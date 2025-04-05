**🚀 Real-Time Network Anomaly Detection using NVIDIA RAPIDS, Autoencoder, and DBSCAN**


📖 Project Description

  This project simulates a real-time cybersecurity pipeline for detecting network anomalies using GPU-accelerated tools from NVIDIA RAPIDS, PyTorch, and     unsupervised ML models. It combines an **autoencoder neural network** for anomaly detection and DBSCAN clustering to identify unusual network traffic patterns. Additionally, it simulates firewall actions, automated takedown mechanisms, and **threat scoring** using a simulated **NVIDIA Morpheus API**.

--------------------------------------------------------------

💻 **Features**

  - ✅ Real-time network traffic monitoring
  - ✅ Autoencoder-based anomaly detection (PyTorch)
  - ✅ DBSCAN clustering (cuML GPU-accelerated)
  - ✅ Simulated NVIDIA Morpheus threat scoring API
  - ✅ Automated firewall actions using NVIDIA DOCA simulation
  - ✅ Simulated takedown action
  - ✅ Synthetic network log generation with masked IPs
  - ✅ CUDA GPU acceleration for large-scale data handling

--------------------------------------------------------------

⚙️ **Tech Stack / Tools / Libraries**

  | **Tool/Library**                    | **Purpose/Usage**                                |
  | ----------------------------------- | ------------------------------------------------ |
  | `cudf`, `cupy`, `cuml` (RAPIDS AI)  | GPU-based data processing, clustering            |
  | `torch` (PyTorch)                   | Autoencoder neural network for anomaly detection |
  | `numpy`                             | Numerical operations                             |
  | `random`, `datetime`, `time`        | Data simulation and handling                     |
  | `nvidia-smi`, `nvcc`                | GPU and CUDA environment checks                  |
  | NVIDIA Morpheus API (Simulated)     | Threat detection scoring                         |
  | NVIDIA DOCA (Simulated)             | Firewall action automation                       |

---------------------------------------------------------------

🏗️ **Architecture Overview**

  ```
  Synthetic Network Data (features + masked IPs)
           |
    [ cuDF DataFrame (GPU) ]
           |
      DBSCAN Clustering (cuML) ------> Outlier detection (label -1)
           |
     Autoencoder Model (PyTorch)
           |
  Reconstruction Error Calculation
           |
  Morpheus Threat Scoring (Simulated API)
           |
  Combined Threat Score Analysis
           |
  Decision:
      - Normal traffic --> Continue
      - High Anomaly Score -->
              |--> NVIDIA DOCA Firewall Action (Simulated)
              |--> Automated Content Takedown (Simulated)
  ```

----------------------------------------------------

🧐 **Algorithms Used**

  | **Section**       | **Algorithm / Model**     | **Why Used?**                                                     |
  | ----------------- | ------------------------- | ----------------------------------------------------------------- |
  | Clustering        | DBSCAN (cuML GPU)         | Detects outliers, non-linear separation, unsupervised             |
  | Anomaly Detection | Autoencoder (PyTorch)     | Learns normal patterns, reconstruction error highlights anomalies |
  | Threat Scoring    | Simulated Morpheus        | Provides external threat assessment                               |
  | Decision Making   | Threshold-based logic     | Automates firewall & takedown                                     |

------------------------------------------------------

📝 **How It Works (Flow)**

  1. **Data Simulation:** Generates synthetic network logs with masked IPs and random features.
  2. **DBSCAN Clustering:** Groups similar traffic, detects noise/outliers as potential anomalies.
  3. **Autoencoder Neural Network:** Reconstructs normal data; high reconstruction error indicates anomaly.
  4. **Morpheus API Simulation:** Adds a secondary simulated threat score.
  5. **Decision Logic:** If score exceeds the threshold:
     - Block IP (simulated DOCA Firewall)
     - Trigger automated takedown.
  6. **Logging:** Outputs sample traffic, anomalies, and actions taken.

---------------------------------------------------------

📊 **Data Details**

  - **Features:** `feature1`, `feature2`, `feature3` represent characteristics of network packets.
  - **Masked IPs:** `src_ip`, `dst_ip` are partially hidden for privacy.
  - **Traffic Protocols:** `TCP`, `UDP` protocols simulate real-world network traffic.

----------------------------------------------------------

🔐 **Cybersecurity Relevance**

  - **Morpheus API (Simulated):** Adds intelligence-based threat analysis.
  - **Firewall Action:** Prevents malicious IPs from further interaction.
  - **Takedown Simulation:** Demonstrates proactive content removal.

-----------------------------------------------------------

🔎 **Commands / Setup**

  ```bash
  # Check GPU and CUDA versions
  nvidia-smi
  nvcc --version

  # Install RAPIDS packages
  pip install cudf-cu12 cuml-cu12 cugraph-cu12 --extra-index-url=https://pypi.ngc.nvidia.com

  # Run the simulation
  python your_script.py
  ```

------------------------------------------------------------

📈 **Accuracy / Security Notes**

  - ✅ Accuracy relies on threshold tuning in:
  - DBSCAN parameters (`eps`, `min_samples`)
  - Autoencoder reconstruction error threshold
  - ✅ Synthetic Data is randomly generated (no real data compromise risk).
  - ✅ Firewall & Takedown logic is simulated; real implementation needs secure API integrations.

-------------------------------------------------------------

🔐 **What is the Morpheus API Key For?**

  - Represents authentication for external threat analysis.
  - Simulates sending data to NVIDIA Morpheus for advanced ML-based threat detection
  - Helps mimic a real-world secure API call flow.

-------------------------------------------------------------

🛠️ **License**

  - MIT

------------------------------------------------------------

🌐 **API References / Used Concepts**

  - Morpheus API (Simulated)
  - NVIDIA DOCA Firewall API (Simulated)
  - DBSCAN (Density-Based Spatial Clustering)
  - Autoencoder (Unsupervised Deep Learning)
  - NVIDIA RAPIDS Libraries: cudf, cuml

-------------------------------------------------------------

✍️ **Author**

  Yogesh J L ,
  Yogesh-J-L(Github Profile)



