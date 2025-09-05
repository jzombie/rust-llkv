[Draft]

- Physical/Logical key separation (batched writes of many logical keys can consume far less physical keys than writing directly to storage keys).
- Logical keys namespaced per field.
- Supports variable and fixed width keys.
- Supports variable and fixed width values.
- Logical key and value segment pruning.
