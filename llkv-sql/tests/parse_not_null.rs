use sqlparser::ast::{BinaryOperator, Expr as SqlExpr, Statement, UnaryOperator};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

#[test]
fn not_null_comparison_parses_as_not_over_comparison() {
    let sql = "SELECT * FROM tab1 AS cor0 CROSS JOIN tab2 WHERE NOT ( NULL ) >= NULL";
    let dialect = GenericDialect {};
    let mut statements = Parser::parse_sql(&dialect, sql).expect("parse");
    assert_eq!(statements.len(), 1, "expected single statement");

    let statement = statements.pop().expect("statement");
    let select = match statement {
        Statement::Query(query) => match *query.body {
            sqlparser::ast::SetExpr::Select(select) => select,
            other => panic!("unexpected query body: {other:?}"),
        },
        other => panic!("unexpected statement kind: {other:?}"),
    };

    let selection = select
        .selection
        .expect("query should include WHERE clause selection");

    match selection {
        SqlExpr::UnaryOp {
            op: UnaryOperator::Not,
            expr,
        } => match expr.as_ref() {
            SqlExpr::BinaryOp {
                op: BinaryOperator::GtEq,
                left,
                right,
            } => {
                assert!(matches!(left.as_ref(), SqlExpr::Nested(_)));
                assert!(matches!(right.as_ref(), SqlExpr::Value(_)));
            }
            other => panic!("unexpected inner expression: {other:?}"),
        },
        other => panic!("unexpected WHERE expression: {other:?}"),
    }
}
