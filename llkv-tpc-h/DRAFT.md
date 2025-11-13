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

---

https://github.com/apache/datafusion-benchmarks?tab=readme-ov-file#tpc-legal-considerations

It is important to know that TPC benchmarks are copyrighted IP of the Transaction Processing Council. Only members of the TPC consortium are allowed to publish TPC benchmark results. Fun fact: only four companies have published official TPC-DS benchmark results so far, and those results can be seen [here](https://www.tpc.org/tpcds/results/tpcds_results5.asp?orderby=dbms&version=3).

However, anyone is welcome to create derivative benchmarks under the TPC's fair use policy, and that is what we are doing here. We do not aim to run a true TPC benchmark (which is a significant endeavor). We are just running the individual queries and recording the timings.

Throughout this document and when talking about these benchmarks, you will see the term "derived from TPC-H" or "derived from TPC-DS". We are required to use this terminology and this is explained in the [fair-use policy (PDF)](https://www.tpc.org/tpc_documents_current_versions/pdf/tpc_fair_use_quick_reference_v1.0.0.pdf).

DataFusion benchmarks are a Non-TPC Benchmark. Any comparison between official TPC Results with non-TPC workloads is prohibited by the TPC.

## TPC Legal Notices

TPC-H is Copyright © 1993-2022 Transaction Processing Performance Council. The full TPC-H specification in PDF format can be found here.

TPC-DS is Copyright © 2021 Transaction Processing Performance Council. The full TPC-DS specification in PDF format can be found here.

TPC, TPC Benchmark, TPC-H, and TPC-DS are trademarks of the Transaction Processing Performance Council.
