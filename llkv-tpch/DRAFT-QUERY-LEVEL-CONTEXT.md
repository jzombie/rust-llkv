# Per-Query Perf Outline

*As documented by "Quantifying TPC-H Choke Points and Their Optimizations"*

## Q4

### References:

- 4.1 Join Ordering
- 4.2 Predicate Pushdown and Ordering
- 4.3 Between Composition
- 4.5 Physical Locality
- 4.7 Flattening Subqueries
- 4.8 Semi Join Reduction
- 4.10 Result Reuse

### Query Characteristics and Structure

- **Correlated Subqueries**: Q4 is one of six TPC-H queries that utilize correlated subqueries. Specifically, it uses correlated (NOT) EXISTS subqueries.
- **Date Range Filtering**: The query filters by a date using `>=` and `<` predicates.
- **Join Complexity**: Q4 is one of only seven join queries that can finish execution within a minute without needing sophisticated join predicate matching steps, whereas most other join queries are virtually unexecutable without them.
- **Parameters** It has 1 parameter and 58 variations.
- **Result Cache Potential**: The query has a result size of 1142 B. A full cache would require 64.7 KB, and the estimated benefit of result caching is 5.39.

### Optimization Choke Points

The document highlights several specific optimizations (choke points) that significantly impact Q4:

1. **Flattening Subqueries (High Impact)**

   - Flattening the correlated subqueries is a critical optimization for Q4.
   - By flattening the subquery, the DBMS avoids executing the subquery once per input row.
   - This optimization results in a performance improvement of approximately 288% for Q4.

2. **Semi Join Reduction (High Relevance)**
   
   - Q4 serves as a specific use case for semi join reductions outside of flattened subqueries.
   - **Context**: The query filters the `lineitem` table for rows where `l_commitdate < l_receiptdate`, which reduces cardinality by less than 40%. It is joined with the orders table, which is filtered on `o_orderdate` where less than 5% of orders qualify.
   - **Mechanism**: Adding a semi join reduction reduces the input to the predicate on the lineitem table based on the filtered orders table.
   - **Impact**: Selectively adding semi joins improves Q4 performance significantly, showing a similar gain to subquery flattening in isolation (approx. 284-288%).

3. **Predicate Pushdown and Ordering**

Q4 benefits notably from predicate pushdown and ordering, showing an improvement of 170%.

4. **Join-Dependent Predicate Duplication**

    - While Q4 is affected by this optimization, the impact is smaller compared to others. It shows a 13% improvement when using optimized predicate duplication.
    - Conversely, unoptimized duplication strategies can lead to a performance regression of -41%.

