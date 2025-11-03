```mermaid
flowchart TD
  A[Start: Define Problem] --> B[Collect Data]
  B --> C[Explore and Visualize]
  C --> D[Preprocess / Clean / Feature Engineering]
  D --> E[Choose Model]
  E --> F[Train Model]
  F --> G[Validate and Tune Hyperparameters]
  G --> H[Evaluate on Test Set]
  H --> I{Good Enough?}
  I -- Yes --> J[Deploy Model]
  I -- No --> K[Go Back: Data / Features / Model]
  J --> L[Monitor and Maintain]
  L --> M[Retrain or Update]
  M --> L

