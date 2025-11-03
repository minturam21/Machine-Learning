```mermaid
flowchart TD
  A[Start: Define Problem] --> B[Collect Data]
  B --> C[Explore & Visualize]
  C --> D[Preprocess / Clean / Feature Eng]
  D --> E[Choose Model(s)]
  E --> F[Train Model]
  F --> G[Validate / Tune Hyperparams]
  G --> H[Evaluate on Test Set]
  H --> I{Good Enough?}
  I -- Yes --> J[Deploy Model]
  I -- No --> K[Go back: Data / Features / Model]
  J --> L[Monitor & Maintain]
  L --> M[Retrain / Update]
  M --> L
