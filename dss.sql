-- REGION
CREATE TABLE region (
  r_regionkey   INTEGER NOT NULL PRIMARY KEY,
  r_name        CHAR(25) NOT NULL,
  r_comment     VARCHAR(152)
);

-- NATION
CREATE TABLE nation (
  n_nationkey   INTEGER NOT NULL PRIMARY KEY,
  n_name        CHAR(25) NOT NULL,
  n_regionkey   INTEGER NOT NULL,
  n_comment     VARCHAR(152),
  FOREIGN KEY (n_regionkey) REFERENCES region(r_regionkey)
);

-- PART
CREATE TABLE part (
  p_partkey     INTEGER NOT NULL PRIMARY KEY,
  p_name        VARCHAR(55) NOT NULL,
  p_mfgr        CHAR(25) NOT NULL,
  p_brand       CHAR(10) NOT NULL,
  p_type        VARCHAR(25) NOT NULL,
  p_size        INTEGER NOT NULL,
  p_container   CHAR(10) NOT NULL,
  p_retailprice DECIMAL(15,2) NOT NULL,
  p_comment     VARCHAR(23) NOT NULL
);

-- SUPPLIER
CREATE TABLE supplier (
  s_suppkey     INTEGER NOT NULL PRIMARY KEY,
  s_name        CHAR(25) NOT NULL,
  s_address     VARCHAR(40) NOT NULL,
  s_nationkey   INTEGER NOT NULL,
  s_phone       CHAR(15) NOT NULL,
  s_acctbal     DECIMAL(15,2) NOT NULL,
  s_comment     VARCHAR(101) NOT NULL,
  FOREIGN KEY (s_nationkey) REFERENCES nation(n_nationkey)
);

-- PARTSUPP
CREATE TABLE partsupp (
  ps_partkey    INTEGER NOT NULL,
  ps_suppkey    INTEGER NOT NULL,
  ps_availqty   INTEGER NOT NULL,
  ps_supplycost DECIMAL(15,2) NOT NULL,
  ps_comment    VARCHAR(199) NOT NULL,
  PRIMARY KEY (ps_partkey, ps_suppkey),
  FOREIGN KEY (ps_partkey) REFERENCES part(p_partkey),
  FOREIGN KEY (ps_suppkey) REFERENCES supplier(s_suppkey)
);

-- CUSTOMER
CREATE TABLE customer (
  c_custkey     INTEGER NOT NULL PRIMARY KEY,
  c_name        VARCHAR(25) NOT NULL,
  c_address     VARCHAR(40) NOT NULL,
  c_nationkey   INTEGER NOT NULL,
  c_phone       CHAR(15) NOT NULL,
  c_acctbal     DECIMAL(15,2) NOT NULL,
  c_mktsegment  CHAR(10) NOT NULL,
  c_comment     VARCHAR(117) NOT NULL,
  FOREIGN KEY (c_nationkey) REFERENCES nation(n_nationkey)
);

-- ORDERS
CREATE TABLE orders (
  o_orderkey      INTEGER NOT NULL PRIMARY KEY,
  o_custkey       INTEGER NOT NULL,
  o_orderstatus   CHAR(1) NOT NULL,
  o_totalprice    DECIMAL(15,2) NOT NULL,
  o_orderdate     DATE NOT NULL,
  o_orderpriority CHAR(15) NOT NULL,
  o_clerk         CHAR(15) NOT NULL,
  o_shippriority  INTEGER NOT NULL,
  o_comment       VARCHAR(79) NOT NULL,
  FOREIGN KEY (o_custkey) REFERENCES customer(c_custkey)
);

-- LINEITEM
CREATE TABLE lineitem (
  l_orderkey     INTEGER NOT NULL,
  l_partkey      INTEGER NOT NULL,
  l_suppkey      INTEGER NOT NULL,
  l_linenumber   INTEGER NOT NULL,
  l_quantity     DECIMAL(15,2) NOT NULL,
  l_extendedprice DECIMAL(15,2) NOT NULL,
  l_discount     DECIMAL(15,2) NOT NULL,
  l_tax          DECIMAL(15,2) NOT NULL,
  l_returnflag   CHAR(1) NOT NULL,
  l_linestatus   CHAR(1) NOT NULL,
  l_shipdate     DATE NOT NULL,
  l_commitdate   DATE NOT NULL,
  l_receiptdate  DATE NOT NULL,
  l_shipinstruct CHAR(25) NOT NULL,
  l_shipmode     CHAR(10) NOT NULL,
  l_comment      VARCHAR(44) NOT NULL,
  PRIMARY KEY (l_orderkey, l_linenumber),
  FOREIGN KEY (l_orderkey) REFERENCES orders(o_orderkey),
  FOREIGN KEY (l_partkey, l_suppkey)
    REFERENCES partsupp(ps_partkey, ps_suppkey)
);

CREATE INDEX idx_orders_custkey ON orders(o_custkey);
CREATE INDEX idx_lineitem_orderkey ON lineitem(l_orderkey);
CREATE INDEX idx_lineitem_part_supp ON lineitem(l_partkey, l_suppkey);
CREATE INDEX idx_partsupp_part ON partsupp(ps_partkey);
CREATE INDEX idx_partsupp_supp ON partsupp(ps_suppkey);
CREATE INDEX idx_customer_nation ON customer(c_nationkey);
CREATE INDEX idx_supplier_nation ON supplier(s_nationkey);
CREATE INDEX idx_nation_region ON nation(n_regionkey);
