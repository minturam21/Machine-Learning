```mermaid
flowchart TD

%% FOUNDATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A[Start: Learn Python] --> B[Math & Statistics Foundation]
B --> B1[Linear Algebra: vectors, matrices, dot product, eigenvalues]
B --> B2[Calculus: derivatives, gradients, chain rule]
B --> B3[Probability & Stats: mean, variance, distribution, Bayes, sampling]
B --> B4[Programming Libraries: NumPy, Pandas, Matplotlib, Seaborn]

%% DATA HANDLING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B4 --> C[Data Preprocessing & EDA]
C --> C1[Data Cleaning: missing values, outliers, duplicates]
C --> C2[Feature Engineering: encoding, scaling, transformations]
C --> C3[EDA: visualization, distributions, correlations]
C --> C4[Train-Test Split, Data Leakage]

%% SUPERVISED LEARNING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C4 --> D[Supervised Learning]

D --> D1[Regression]
D1 --> D1a[Linear Regression]
D1a --> D1b[Multiple Linear Regression]
D1b --> D1c[Polynomial Regression]
D1c --> D1d[Ridge & Lasso Regression]
D1d --> D1e[Elastic Net]
D1e --> D1f[Regularization Concepts]
D1f --> D1g[Evaluation Metrics: RMSE, MAE, R²]

D --> D2[Classification]
D2 --> D2a[Logistic Regression]
D2a --> D2b[K-Nearest Neighbors (KNN)]
D2b --> D2c[Support Vector Machine (SVM)]
D2c --> D2d[Decision Tree Classifier]
D2d --> D2e[Random Forest Classifier]
D2e --> D2f[XGBoost / LightGBM / CatBoost]
D2f --> D2g[Naive Bayes]
D2g --> D2h[Evaluation Metrics: Accuracy, Precision, Recall, F1, AUC]

D --> D3[Model Validation]
D3 --> D3a[Cross-Validation]
D3a --> D3b[Bias-Variance Tradeoff]
D3b --> D3c[Hyperparameter Tuning: GridSearchCV, RandomizedSearchCV]

%% UNSUPERVISED LEARNING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D3c --> E[Unsupervised Learning]
E --> E1[Clustering Algorithms]
E1 --> E1a[K-Means]
E1a --> E1b[Hierarchical Clustering]
E1b --> E1c[DBSCAN]
E --> E2[Dimensionality Reduction]
E2 --> E2a[PCA (Principal Component Analysis)]
E2a --> E2b[t-SNE, UMAP]
E2b --> E2c[Feature Selection Methods]

%% ENSEMBLE METHODS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
E2c --> F[Ensemble Methods]
F --> F1[Bagging]
F1 --> F2[Boosting]
F2 --> F3[Stacking]
F3 --> F4[Random Forest, AdaBoost, Gradient Boosting, XGBoost]

%% MODEL OPTIMIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F4 --> G[Model Optimization]
G --> G1[Regularization: L1/L2]
G --> G2[Feature Importance & Selection]
G --> G3[Learning Curves & Overfitting Control]

%% DEEP LEARNING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
G3 --> H[Deep Learning]
H --> H1[Neural Network Basics: Perceptron, Forward & Backpropagation]
H1 --> H2[Activation Functions: ReLU, Sigmoid, Tanh, Softmax]
H2 --> H3[Loss Functions: MSE, Cross-Entropy, Hinge Loss]
H3 --> H4[Optimizers: SGD, Adam, RMSprop]
H4 --> H5[CNN (Convolutional Neural Networks)]
H5 --> H6[RNN & LSTM (Sequential Data)]
H6 --> H7[Autoencoders, Transfer Learning, Attention]
H7 --> H8[Frameworks: TensorFlow, Keras, PyTorch]

%% ADVANCED TOPICS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H8 --> I[Advanced Machine Learning Topics]
I --> I1[NLP (Natural Language Processing)]
I1 --> I1a[Text Preprocessing: tokenization, stopwords, stemming, lemmatization]
I1a --> I1b[Word Embeddings: Word2Vec, GloVe, BERT]
I --> I2[Reinforcement Learning]
I2 --> I2a[Q-Learning, Policy Gradients, DQN]
I --> I3[Generative AI]
I3 --> I3a[GANs, VAEs, Diffusion Models]
I3a --> I3b[Prompt Engineering, LLMs (ChatGPT, Gemini, etc.)]

%% DEPLOYMENT & MLOPS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I3b --> J[Model Deployment & MLOps]
J --> J1[Export Model: Pickle, Joblib, ONNX]
J1 --> J2[Build APIs: Flask, FastAPI]
J2 --> J3[Containerization: Docker]
J3 --> J4[Cloud: AWS, GCP, Azure]
J4 --> J5[MLOps: CI/CD, Model Monitoring, Data Drift, Retraining]
J5 --> K[End of Learning Path 🚀]
