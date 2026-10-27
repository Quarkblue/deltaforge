#pragma once

// Generic printing for any safetensors file: walks the tensor map, so it knows nothing
// about layers, vision blocks or MTP. Lives outside SafeTensors because the loader itself
// prints nothing (NFR-6); the caller chooses the stream.

#include "safetensors.hpp"

#include <map>
#include <ostream>
#include <string>

namespace manifest {

// "[30, 8, 256]"; a scalar (ndim 0) prints "[]"
inline std::string shapeToString(const TensorView& tv) {
    std::string s = "[";
    for (int i = 0; i < tv.ndim; ++i) {
        if (i) s += ", ";
        s += std::to_string(tv.shape[i]);
    }
    return s + "]";
}

// One tensor, one tab-separated line: name, shape, dtype. Tabs (not aligned columns) so the
// output diffs cleanly against the Python manifest.
inline void printTensor(std::ostream& os, const std::string& name, const TensorView& tv) {
    os << name << '\t' << shapeToString(tv) << '\t' << dtype_to_string(tv.dtype) << '\n';
}

// Pointer overload for typed views (Model), where a field may be unresolved.
inline void printTensor(std::ostream& os, const std::string& label, const TensorView* tv) {
    if (!tv) { os << label << "\t<null>\n"; return; }
    printTensor(os, label, *tv);
}

// Every tensor whose name starts with `prefix` (empty = all), in sorted order.
// e.g. prefix "L03." for one dumped layer, "model.language_model.layers.3." for one model layer.
inline void printManifest(std::ostream& os, const SafeTensors& st, const std::string& prefix = "") {
    for (const std::string& name : st.sortedNames()) {
        if (name.compare(0, prefix.size(), prefix) != 0) continue;
        printTensor(os, name, st.at(name));
    }
}

// "196 tensors, 51510128 bytes | BF16 177, F32 18, I64 1"
inline void printSummary(std::ostream& os, const SafeTensors& st) {
    size_t total_bytes = 0;
    std::map<std::string, size_t> per_dtype;  // std::map: sorted, so the line is stable
    for (const auto& [name, tv] : st.getTensors()) {
        total_bytes += tv.nbytes;
        per_dtype[dtype_to_string(tv.dtype)]++;
    }
    os << st.getTensors().size() << " tensors, " << total_bytes << " bytes | ";
    bool first = true;
    for (const auto& [dtype, count] : per_dtype) {
        os << (first ? "" : ", ") << dtype << ' ' << count;
        first = false;
    }
    os << '\n';
}

// The file's __metadata__ (string -> string), sorted by key. Prints nothing if absent.
inline void printMetadata(std::ostream& os, const SafeTensors& st) {
    for (const auto& [key, value] : st.metadata().items()) {
        os << key << '\t' << (value.is_string() ? value.get<std::string>() : value.dump()) << '\n';
    }
}

}  // namespace manifest
