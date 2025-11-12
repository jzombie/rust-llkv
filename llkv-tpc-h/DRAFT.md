https://www.tpc.org/tpch/

Downloads: https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp
TPC-H Spec: https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPC-H_v3.0.1.pdf

The TPC Tools are available free of charge, however all users must agree to the licensing terms and register prior to use.
TPC-H Tools: https://www.tpc.org/TPC_Documents_Current_Versions/download_programs/tools-download-request5.asp?bm_type=TPC-H&bm_vers=3.0.1&mode=CURRENT-ONLY

---

#####
Personal notes
#####

Two available modes:

1. Accuracy (nothing else matters if the results are not accurate)
2. Performance

Section 2.3 of the spec, "Query Validation," may be of interest for point 1.

---

Typical flow:

1. Run `dbgen/dss.ddl` to create the 8 TPC-H tables.
2. Run `dbgen/dss.ri` to add PKs and FKs.
3. Use `dbgen` to generate .tbl data and load it into those tables.

Related files:

- `dbgen/dists.dss` — data distributions.
- `dbgen/dss.h` — column widths/types used by dbgen.

---

https://www.youtube.com/watch?v=xotGodf7gZY - "If you're going to compare two databases [or solutions] against each other, what should be the [benchmarking] criteria used for testing?

TPC goals:

- Verify existing benchmarks (requires an auditor)
- Creating good benchmarks (everyone can use them, even without an auditor)
- Creating a good process for reviewing and monitoring those benchmarks
- Resolution of disputes and challenges
- Good benchmarks are like good laws. They lay the foundation for civilized (fair) competition.
