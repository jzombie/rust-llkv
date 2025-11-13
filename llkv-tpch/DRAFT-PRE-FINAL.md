Note: This is a preemptive summary and does not yet incorporate all of the context from the draft.

---

# What TPC-H is

* A decision-support benchmark built around **22 ad-hoc SQL queries** plus **two refresh functions** (RF1/RF2) over an 8-table schema (customer, lineitem, orders, partsupp, part, supplier, nation, region). It reports performance in **QphH** (queries per hour for TPC-H). ([TPC][1])
* The spec defines schema, data types, distributions, query text and rules, **ACID** requirements, scaling rules, and execution/metric rules. ([TPC][2])

# Two phases you’ll care about

1. **Accuracy (functional compliance)**
   You must run queries exactly per spec and **validate answers** per the Query Validation rules before any performance numbers matter. See **Section 2.3 “Query Validation.”** ([TPC][2])

2. **Performance**
   After functional validation, you measure **Power** (single stream) and **Throughput** (multiple parallel query streams). TPC-H combines them into **QphH@Size**, with reporting/price rules also defined in the spec. ([TPC][2])

# Practical “getting started” flow

1. **Create the schema:** run the TPC tools’ DDL (commonly `dbgen/dss.ddl`).
2. **Add constraints:** run `dbgen/dss.ri` to set PKs/FKs.
3. **Generate data:** use **DBGEN** to produce `.tbl` files at your chosen **Scale Factor** (e.g., SF1, SF10, SF1000). Section **4.2** covers DBGEN and population. ([TPC][2])
4. **Load data:** bulk-load `.tbl` into your DB.
5. **Generate query text:** use **QGEN** to materialize the 22 query variants. Section **2.x** covers query requirements and substitutions. ([TPC][2])
6. **Validate answers:** follow **Section 2.3**; compare to the answer sets per spec rules. ([TPC][2])
7. **Run performance:** execute per **Section 5** (Power and Throughput), then compute/report **QphH** and disclosures as required. ([TPC][2])

# Files you’ll touch in the tools

* `dbgen/dss.ddl` — table creation DDL.
* `dbgen/dss.ri` — constraints (PK/FK).
* `dbgen/dists.dss` — data distributions used by DBGEN.
* `dbgen/dss.h` — column widths/types for generation.
  (Names are from the TPC-H toolset; see the spec’s data generation and the official download page below.) ([TPC][2])

# Tools, spec, and licensing

* **Spec (v3.0.1)** is public; it defines everything you need (schema, queries, ACID, execution rules, metrics). ([TPC][2])
* **TPC-H tools (DBGEN/QGEN)** are **free**, but require acceptance of the license and **registration** before download. Use the **“Download Programs”** entry for TPC-H on the TPC site. ([TPC][3])

# Key reminders

* The 22 queries are simple to list but **hard to implement correctly** because the spec constrains SQL features, substitution parameters, null/rounding behavior, and requires ACID and strict validation before performance claims count. Sections **2 (queries)**, **3 (ACID)**, **4 (scaling/population)**, and **5 (metrics/execution)** are the core. ([TPC][2])

If you want, I can script a small harness outline that: calls DBGEN, loads data, runs QGEN output, performs Section 2.3 checks, and times Power/Throughput runs with clean logs aligned to the spec.

[1]: https://www.tpc.org/tpch/ "TPC-H Homepage"
[2]: https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPC-H_v3.0.1.pdf "0: INTRODUCTION"
[3]: https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp "TPC Current Specs"
