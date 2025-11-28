use arrow::array::{Decimal128Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use llkv_aggregate::{AggregateAccumulator, AggregateKind, AggregateSpec};
use std::sync::Arc;

#[test]
fn test_avg_decimal128_rounding() {
    // Create a batch with Decimal128 values
    // 10.51 and 10.52. Sum = 21.03. Count = 2. Avg = 10.515.
    // If truncated: 10.51.
    // If rounded: 10.52.

    // Precision 10, Scale 2.
    // 10.51 -> 1051
    // 10.52 -> 1052

    let array = Decimal128Array::from(vec![1051, 1052])
        .with_precision_and_scale(10, 2)
        .unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Decimal128(10, 2),
        true,
    )]));

    let batch = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();

    let spec = AggregateSpec {
        kind: AggregateKind::Avg {
            field_id: 0,
            data_type: DataType::Decimal128(10, 2),
            distinct: false,
        },
        alias: "avg".to_string(),
    };

    let mut accumulator =
        AggregateAccumulator::new_with_projection_index(&spec, Some(0), None).unwrap();
    accumulator.update(&batch).unwrap();

    let (_field, result_array) = accumulator.finalize().unwrap();

    let decimal_array = result_array
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    let value = decimal_array.value(0);

    // Expected: 1052 (10.52) because 10.515 rounds up (half away from zero).
    assert_eq!(value, 1052, "Expected 10.52, got {}", value);
}
