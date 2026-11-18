// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.

//
// exrrepair - rebuild a damaged OpenEXR chunk offset table.
//
// When a disk write goes bad, the chunk offset table of an EXR file can
// end up zeroed, truncated or filled with garbage. exrcheck diagnoses
// this but cannot fix it, forcing a full re-render. This tool scans the
// chunk records of every part, reconstructs the offset table and writes
// a repaired file:
//
//  - chunks are processed one at a time (streamed); the whole image is
//    never resident in memory;
//  - chunk payloads are copied verbatim, so pixels are bit-identical
//    to the original;
//  - a file whose offset table turns out to be intact is passed through
//    byte-for-byte;
//  - every chunk is decompressed once to prove the repaired file will
//    pass exrcheck;
//  - if the file cannot be repaired, a distinct non-zero exit code is
//    returned instead of a silent success.
//
// Exit codes:
//   0  success (file was intact and copied through, or repair succeeded)
//   1  usage error
//   2  input cannot be opened or header is corrupt (not repairable)
//   3  chunk records missing/corrupt: offset table cannot be rebuilt
//   4  chunk data found but does not decode (not repairable)
//   5  output i/o error
//

#include "openexr.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int EXIT_OK              = 0;
constexpr int EXIT_USAGE           = 1;
constexpr int EXIT_BAD_INPUT       = 2;
constexpr int EXIT_SCAN_FAILED     = 3;
constexpr int EXIT_VALIDATE_FAILED = 4;
constexpr int EXIT_IO_ERROR        = 5;

bool        g_verbose = false;
std::string g_lastError;

void
quiet_error_handler (exr_const_context_t, exr_result_t, const char* msg)
{
    // keep the first message, it is usually the most specific
    if (g_lastError.empty () && msg) g_lastError = msg;
}

void
reset_error () { g_lastError.clear (); }

//
// Read stream wrapper that serves zeroes for the byte ranges holding the
// on-disk chunk offset tables. Opening a context through this forces the
// core library to ignore the (possibly damaged) table and rebuild it by
// scanning the chunk records, which is exactly the repair we need.
//
struct ZeroingStream
{
    std::ifstream                            fs;
    std::mutex                               mtx;
    int64_t                                  size = 0;
    std::vector<std::pair<uint64_t, uint64_t>> regions; // [start, end)
};

int64_t
zeroing_read (
    exr_const_context_t         ctxt,
    void*                       userdata,
    void*                       buffer,
    uint64_t                    sz,
    uint64_t                    offset,
    exr_stream_error_func_ptr_t error_cb)
{
    ZeroingStream* zs = static_cast<ZeroingStream*> (userdata);
    std::lock_guard<std::mutex> lk (zs->mtx);

    if (offset >= (uint64_t) zs->size) return 0;
    uint64_t avail = (uint64_t) zs->size - offset;
    if (sz > avail) sz = avail;

    zs->fs.clear ();
    zs->fs.seekg ((std::streamoff) offset, std::ios::beg);
    zs->fs.read ((char*) buffer, (std::streamsize) sz);
    int64_t n = (int64_t) zs->fs.gcount ();
    if (n <= 0) return zs->fs.bad () ? -1 : 0;

    for (const auto& r : zs->regions)
    {
        uint64_t b = std::max (offset, r.first);
        uint64_t e = std::min (offset + (uint64_t) n, r.second);
        if (b < e) memset ((char*) buffer + (b - offset), 0, e - b);
    }
    return n;
}

int64_t
zeroing_size (exr_const_context_t, void* userdata)
{
    return static_cast<ZeroingStream*> (userdata)->size;
}

struct ChunkRec
{
    uint64_t offset = 0;
    uint64_t length = 0;
};

struct PartInfo
{
    exr_storage_t         storage = EXR_STORAGE_SCANLINE;
    int32_t               chunk_count = 0;
    uint64_t              table_offset = 0;
    exr_attr_box2i_t      dw = {{0, 0}, {0, 0}};
    int32_t               lines_per_chunk = 0;
    int32_t               levels_x = 0, levels_y = 0;
    exr_tile_level_mode_t level_mode = EXR_TILE_ONE_LEVEL;
    std::vector<ChunkRec> chunks;
};

enum class AnalyzeResult
{
    ok,
    table_fail,
    decode_fail
};

bool
gather_part_info (exr_const_context_t ctxt, std::vector<PartInfo>& parts)
{
    int nparts = 0;
    if (exr_get_count (ctxt, &nparts) != EXR_ERR_SUCCESS || nparts <= 0)
        return false;

    parts.resize ((size_t) nparts);
    for (int p = 0; p < nparts; ++p)
    {
        PartInfo& pi = parts[(size_t) p];
        if (exr_get_storage (ctxt, p, &pi.storage) != EXR_ERR_SUCCESS)
            return false;
        if (exr_get_chunk_count (ctxt, p, &pi.chunk_count) != EXR_ERR_SUCCESS)
            return false;
        if (exr_get_chunk_table_offset (ctxt, p, &pi.table_offset) !=
            EXR_ERR_SUCCESS)
            return false;
        if (exr_get_data_window (ctxt, p, &pi.dw) != EXR_ERR_SUCCESS)
            return false;

        if (pi.storage == EXR_STORAGE_SCANLINE ||
            pi.storage == EXR_STORAGE_DEEP_SCANLINE)
        {
            if (exr_get_scanlines_per_chunk (ctxt, p, &pi.lines_per_chunk) !=
                EXR_ERR_SUCCESS)
                return false;
        }
        else
        {
            uint32_t             tx = 0, ty = 0;
            exr_tile_round_mode_t round;
            if (exr_get_tile_descriptor (
                    ctxt, p, &tx, &ty, &pi.level_mode, &round) !=
                EXR_ERR_SUCCESS)
                return false;
            if (exr_get_tile_levels (ctxt, p, &pi.levels_x, &pi.levels_y) !=
                EXR_ERR_SUCCESS)
                return false;
        }
        pi.chunks.assign ((size_t) pi.chunk_count, ChunkRec{});
    }
    return true;
}

exr_result_t
validate_chunk_decode (
    exr_const_context_t ctxt, int part, const exr_chunk_info_t* cinfo)
{
    exr_decode_pipeline_t dec = EXR_DECODE_PIPELINE_INITIALIZER;
    exr_result_t          rv  = exr_decoding_initialize (ctxt, part, cinfo, &dec);
    if (rv == EXR_ERR_SUCCESS)
        rv = exr_decoding_choose_default_routines (ctxt, part, &dec);
    if (rv == EXR_ERR_SUCCESS) rv = exr_decoding_run (ctxt, part, &dec);
    exr_decoding_destroy (ctxt, &dec);
    return rv;
}

//
// Walk every chunk of every part in chunk-index order, ask the core
// library for its chunk record (which validates the on-disk chunk leader
// against the expected coordinates) and record the byte span of the
// record. Optionally decompress each chunk to prove it is decodable.
//
AnalyzeResult
analyze (
    exr_const_context_t   ctxt,
    std::vector<PartInfo>& parts,
    uint64_t              file_size,
    bool                  do_decode)
{
    for (size_t p = 0; p < parts.size (); ++p)
    {
        PartInfo& pi = parts[p];

        uint64_t* table = nullptr;
        int32_t   count = 0;
        if (exr_get_chunk_table (ctxt, (int) p, &table, &count) !=
                EXR_ERR_SUCCESS ||
            table == nullptr || count != pi.chunk_count)
            return AnalyzeResult::table_fail;

        int32_t idx = 0;
        auto    record_chunk = [&] (const exr_chunk_info_t& cinfo)
        {
            uint64_t rec_start = table[idx];
            uint64_t rec_end   = cinfo.data_offset + cinfo.packed_size;
            if (rec_start >= rec_end || rec_end > file_size)
                return AnalyzeResult::table_fail;
            pi.chunks[(size_t) idx].offset = rec_start;
            pi.chunks[(size_t) idx].length = rec_end - rec_start;
            if (do_decode &&
                validate_chunk_decode (ctxt, (int) p, &cinfo) !=
                    EXR_ERR_SUCCESS)
                return AnalyzeResult::decode_fail;
            ++idx;
            return AnalyzeResult::ok;
        };

        if (pi.storage == EXR_STORAGE_SCANLINE ||
            pi.storage == EXR_STORAGE_DEEP_SCANLINE)
        {
            for (int32_t i = 0; i < pi.chunk_count; ++i)
            {
                int64_t y =
                    (int64_t) pi.dw.min.y + (int64_t) i * pi.lines_per_chunk;
                exr_chunk_info_t cinfo;
                if (exr_read_scanline_chunk_info (
                        ctxt, (int) p, (int) y, &cinfo) != EXR_ERR_SUCCESS)
                    return AnalyzeResult::table_fail;
                AnalyzeResult r = record_chunk (cinfo);
                if (r != AnalyzeResult::ok) return r;
            }
        }
        else
        {
            // enumerate levels in the exact order chunk indices increase
            std::vector<std::pair<int32_t, int32_t>> levels;
            if (pi.level_mode == EXR_TILE_ONE_LEVEL)
                levels.emplace_back (0, 0);
            else if (pi.level_mode == EXR_TILE_MIPMAP_LEVELS)
                for (int32_t l = 0; l < pi.levels_x; ++l)
                    levels.emplace_back (l, l);
            else // RIPMAP
                for (int32_t ly = 0; ly < pi.levels_y; ++ly)
                    for (int32_t lx = 0; lx < pi.levels_x; ++lx)
                        levels.emplace_back (lx, ly);

            for (const auto& lvl : levels)
            {
                int32_t nx = 0, ny = 0;
                if (exr_get_tile_counts (
                        ctxt, (int) p, lvl.first, lvl.second, &nx, &ny) !=
                    EXR_ERR_SUCCESS)
                    return AnalyzeResult::table_fail;
                for (int32_t ty = 0; ty < ny; ++ty)
                {
                    for (int32_t tx = 0; tx < nx; ++tx)
                    {
                        exr_chunk_info_t cinfo;
                        if (exr_read_tile_chunk_info (
                                ctxt,
                                (int) p,
                                tx,
                                ty,
                                lvl.first,
                                lvl.second,
                                &cinfo) != EXR_ERR_SUCCESS)
                            return AnalyzeResult::table_fail;
                        AnalyzeResult r = record_chunk (cinfo);
                        if (r != AnalyzeResult::ok) return r;
                    }
                }
            }
        }
        if (idx != pi.chunk_count) return AnalyzeResult::table_fail;
    }
    return AnalyzeResult::ok;
}

uint64_t
get_le64 (const uint8_t* p)
{
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

void
put_le64 (uint8_t* p, uint64_t v)
{
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t) (v >> (8 * i));
}

// Read the on-disk offset tables and compare against the chunk record
// offsets found by scanning. Returns false if the table cannot be read
// (e.g. file truncated inside the table) or any entry disagrees.
bool
disk_tables_match (
    const std::string&          filename,
    const std::vector<PartInfo>& parts)
{
    std::ifstream in (filename, std::ios::binary);
    if (!in) return false;
    std::vector<uint8_t> buf;
    for (const auto& pi : parts)
    {
        size_t bytes = (size_t) pi.chunk_count * 8;
        buf.assign (bytes, 0);
        in.seekg ((std::streamoff) pi.table_offset, std::ios::beg);
        in.read ((char*) buf.data (), (std::streamsize) bytes);
        if ((size_t) in.gcount () != bytes) return false;
        for (int32_t i = 0; i < pi.chunk_count; ++i)
            if (get_le64 (buf.data () + (size_t) i * 8) !=
                pi.chunks[(size_t) i].offset)
                return false;
    }
    return true;
}

bool
copy_range (
    std::ifstream& in,
    std::ofstream& out,
    uint64_t       offset,
    uint64_t       length,
    std::vector<char>& buf)
{
    in.clear ();
    in.seekg ((std::streamoff) offset, std::ios::beg);
    uint64_t left = length;
    while (left > 0)
    {
        size_t want = (size_t) std::min<uint64_t> (left, buf.size ());
        in.read (buf.data (), (std::streamsize) want);
        if ((size_t) in.gcount () != want) return false;
        out.write (buf.data (), (std::streamsize) want);
        if (!out) return false;
        left -= want;
    }
    return true;
}

bool
copy_file (const std::string& from, const std::string& to, uint64_t size)
{
    std::ifstream in (from, std::ios::binary);
    std::ofstream out (to, std::ios::binary | std::ios::trunc);
    if (!in || !out) return false;
    std::vector<char> buf (1 << 20);
    bool ok = copy_range (in, out, 0, size, buf);
    out.close ();
    return ok && !!out;
}

bool
write_repaired (
    const std::string&           infile,
    const std::string&           outfile,
    const std::vector<PartInfo>& parts)
{
    uint64_t header_end = UINT64_MAX;
    uint64_t tables     = 0;
    for (const auto& pi : parts)
    {
        header_end = std::min (header_end, pi.table_offset);
        tables += (uint64_t) pi.chunk_count * 8;
    }

    std::ifstream in (infile, std::ios::binary);
    std::ofstream out (outfile, std::ios::binary | std::ios::trunc);
    if (!in || !out) return false;

    std::vector<char> buf (1 << 20);

    // header, verbatim
    if (!copy_range (in, out, 0, header_end, buf)) return false;

    // new offset tables, chunks laid out per part in chunk-index order
    uint64_t running = header_end + tables;
    for (const auto& pi : parts)
    {
        std::vector<uint8_t> tbl ((size_t) pi.chunk_count * 8);
        for (int32_t i = 0; i < pi.chunk_count; ++i)
        {
            put_le64 (tbl.data () + (size_t) i * 8, running);
            running += pi.chunks[(size_t) i].length;
        }
        out.write ((const char*) tbl.data (), (std::streamsize) tbl.size ());
        if (!out) return false;
    }

    // chunk records, verbatim
    for (const auto& pi : parts)
        for (const auto& c : pi.chunks)
            if (!copy_range (in, out, c.offset, c.length, buf)) return false;

    out.close ();
    return !!out;
}

uint64_t
file_size_of (const std::string& filename, bool& ok)
{
    std::ifstream in (filename, std::ios::binary | std::ios::ate);
    if (!in)
    {
        ok = false;
        return 0;
    }
    ok = true;
    return (uint64_t) in.tellg ();
}

exr_context_t
open_normal (const std::string& filename)
{
    exr_context_t             ctxt  = nullptr;
    exr_context_initializer_t cinit = EXR_DEFAULT_CONTEXT_INITIALIZER;
    cinit.error_handler_fn          = &quiet_error_handler;
    if (exr_start_read (&ctxt, filename.c_str (), &cinit) != EXR_ERR_SUCCESS)
        return nullptr;
    return ctxt;
}

exr_context_t
open_zeroing (const std::string& filename, ZeroingStream& zs)
{
    exr_context_t             ctxt  = nullptr;
    exr_context_initializer_t cinit = EXR_DEFAULT_CONTEXT_INITIALIZER;
    cinit.error_handler_fn          = &quiet_error_handler;
    cinit.user_data                 = &zs;
    cinit.read_fn                   = &zeroing_read;
    cinit.size_fn                   = &zeroing_size;
    if (exr_start_read (&ctxt, filename.c_str (), &cinit) != EXR_ERR_SUCCESS)
        return nullptr;
    return ctxt;
}

void
usage (std::ostream& os, const char* prog, bool verbose = false)
{
    os << "Usage: " << prog << " [options] <input.exr> <output.exr>\n";
    if (verbose)
        os << "\n"
              "Rebuild a damaged OpenEXR chunk offset table. The chunk records\n"
              "of every part are scanned, the offset table is rebuilt and a\n"
              "repaired file is written, streaming chunk by chunk. Chunk\n"
              "payloads are copied verbatim so pixels are bit-identical.\n"
              "A file whose table is intact is copied through byte-for-byte.\n"
              "\n"
              "Options:\n"
              "  -v            verbose output\n"
              "  -h, --help    print this message\n"
              "      --version print version information\n"
              "\n"
              "Exit codes:\n"
              "  0  success (intact file copied through, or repair succeeded)\n"
              "  1  usage error\n"
              "  2  input cannot be opened or header is corrupt (not repairable)\n"
              "  3  chunk records missing/corrupt: table cannot be rebuilt\n"
              "  4  chunk data found but does not decode (not repairable)\n"
              "  5  output i/o error\n"
              "\n"
              "Report bugs via https://github.com/AcademySoftwareFoundation/openexr/issues\n";
}

} // namespace

int
main (int argc, char* argv[])
{
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "-v")
            g_verbose = true;
        else if (a == "-h" || a == "--help")
        {
            usage (std::cout, argv[0], true);
            return EXIT_OK;
        }
        else if (a == "--version")
        {
            std::cout << argv[0] << " (OpenEXR " << OPENEXR_VERSION_STRING
                      << ")\n";
            return EXIT_OK;
        }
        else if (!a.empty () && a[0] == '-')
        {
            std::cerr << "unknown option '" << a << "'\n";
            usage (std::cerr, argv[0]);
            return EXIT_USAGE;
        }
        else
            positional.push_back (a);
    }

    if (positional.size () != 2)
    {
        usage (std::cerr, argv[0]);
        return EXIT_USAGE;
    }

    const std::string infile  = positional[0];
    const std::string outfile = positional[1];
    if (infile == outfile)
    {
        std::cerr << "error: input and output paths must differ\n";
        return EXIT_USAGE;
    }

    bool     ok       = false;
    uint64_t fsize    = file_size_of (infile, ok);
    if (!ok)
    {
        std::cerr << "error: cannot open input '" << infile << "'\n";
        return EXIT_BAD_INPUT;
    }

    // Phase 1: parse the header. If the header itself is damaged there
    // is nothing to anchor a repair to.
    reset_error ();
    exr_context_t ctxt = open_normal (infile);
    if (!ctxt)
    {
        std::cerr << "error: cannot parse header of '" << infile
                  << "': " << g_lastError << "\n";
        return EXIT_BAD_INPUT;
    }

    std::vector<PartInfo> parts;
    if (!gather_part_info (ctxt, parts))
    {
        std::cerr << "error: cannot read part structure of '" << infile
                  << "': " << g_lastError << "\n";
        exr_finish (&ctxt);
        return EXIT_BAD_INPUT;
    }

    if (g_verbose)
    {
        for (size_t p = 0; p < parts.size (); ++p)
            std::cerr << "part " << p << ": chunks=" << parts[p].chunk_count
                      << " table_offset=" << parts[p].table_offset
                      << " storage=" << (int) parts[p].storage << "\n";
    }

    // Phase 2: rebuild the table by scanning chunk records. The zeroing
    // stream hides the on-disk table from the library so it must
    // reconstruct it from the chunk data alone.
    ZeroingStream zs;
    zs.fs.open (infile, std::ios::binary);
    zs.size = (int64_t) fsize;
    for (const auto& pi : parts)
        zs.regions.emplace_back (
            pi.table_offset, pi.table_offset + (uint64_t) pi.chunk_count * 8);

    reset_error ();
    exr_context_t  zctxt = open_zeroing (infile, zs);
    AnalyzeResult  scan  = AnalyzeResult::table_fail;
    if (zctxt)
    {
        scan = analyze (zctxt, parts, fsize, /*do_decode=*/true);
        exr_finish (&zctxt);
    }

    if (scan != AnalyzeResult::ok)
    {
        // Fallback: the file may have a valid table but a non-canonical
        // layout (gaps between chunks) that pure scanning cannot handle.
        // Trust the on-disk table once, fully validating every chunk.
        reset_error ();
        AnalyzeResult trust = analyze (ctxt, parts, fsize, /*do_decode=*/true);
        if (trust == AnalyzeResult::ok && disk_tables_match (infile, parts))
        {
            if (!copy_file (infile, outfile, fsize))
            {
                std::cerr << "error: failed writing '" << outfile << "'\n";
                exr_finish (&ctxt);
                return EXIT_IO_ERROR;
            }
            std::cerr << infile << ": offset table intact, copied through\n";
            exr_finish (&ctxt);
            return EXIT_OK;
        }

        bool decode_problem =
            (scan == AnalyzeResult::decode_fail ||
             trust == AnalyzeResult::decode_fail);
        std::cerr << "error: '" << infile << "' is not repairable: "
                  << (decode_problem ? "chunk data does not decode"
                                     : "chunk records missing or corrupt")
                  << " (" << g_lastError << ")\n";
        exr_finish (&ctxt);
        return decode_problem ? EXIT_VALIDATE_FAILED : EXIT_SCAN_FAILED;
    }

    // Phase 3: if the on-disk table already agrees with the scan, the
    // file is intact: pass it through byte-for-byte.
    if (disk_tables_match (infile, parts))
    {
        if (!copy_file (infile, outfile, fsize))
        {
            std::cerr << "error: failed writing '" << outfile << "'\n";
            exr_finish (&ctxt);
            return EXIT_IO_ERROR;
        }
        std::cerr << infile << ": offset table intact, copied through\n";
        exr_finish (&ctxt);
        return EXIT_OK;
    }

    // Phase 4: write the repaired file and re-validate it.
    if (!write_repaired (infile, outfile, parts))
    {
        std::cerr << "error: failed writing repaired file '" << outfile
                  << "'\n";
        exr_finish (&ctxt);
        return EXIT_IO_ERROR;
    }

    reset_error ();
    exr_context_t vctxt = open_normal (outfile);
    if (!vctxt)
    {
        std::cerr << "error: repaired file failed to re-open: " << g_lastError
                  << "\n";
        exr_finish (&ctxt);
        return EXIT_IO_ERROR;
    }
    bool     outsize_ok = false;
    uint64_t outsize    = file_size_of (outfile, outsize_ok);
    std::vector<PartInfo> vparts;
    AnalyzeResult verify = AnalyzeResult::table_fail;
    if (outsize_ok && gather_part_info (vctxt, vparts))
        verify = analyze (vctxt, vparts, outsize, /*do_decode=*/false);
    exr_finish (&vctxt);
    if (verify != AnalyzeResult::ok)
    {
        std::cerr << "error: repaired file failed verification: "
                  << g_lastError << "\n";
        exr_finish (&ctxt);
        return EXIT_IO_ERROR;
    }

    uint64_t total_chunks = 0;
    for (const auto& pi : parts) total_chunks += (uint64_t) pi.chunk_count;
    std::cerr << infile << ": repaired chunk offset table (" << total_chunks
              << " chunks, " << parts.size () << " part(s)) -> " << outfile
              << "\n";
    exr_finish (&ctxt);
    return EXIT_OK;
}
