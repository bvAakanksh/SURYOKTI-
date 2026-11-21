# SURYOKTI: Solar Physics Equations & Evaluation Metrics

This document provides the reference mathematical formulations for all feature engineering, solar position calculations, and regression/anomaly evaluation metrics used throughout the SURYOKTI codebase.

---

## 1. Solar Physics & Position Encodings

### Effective Irradiance on Aperture
$$\text{Effective\_GHI} = \text{GHI} \cdot \max\left(0, \cos(\theta_{\text{SZ}})\right)$$
where $\theta_{\text{SZ}}$ is the Solar Zenith Angle in radians.

### Thermal Stress & Cell Heating Proxy
$$\text{Thermal\_Stress} = \frac{\text{GHI} \cdot T_{\text{ambient}}}{v_{\text{wind}} + 1.0}$$
Models convective cooling by wind against thermal derating caused by solar irradiance absorption.

### Diurnal & Seasonal Cyclical Encodings
$$\text{Hour}_{\sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{Hour}_{\cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$
$$\text{Month}_{\sin} = \sin\left(\frac{2\pi \cdot m}{12}\right), \quad \text{Month}_{\cos} = \cos\left(\frac{2\pi \cdot m}{12}\right)$$
$$\text{WindDir}_{\sin} = \sin\left(\theta_{\text{wind}}\right), \quad \text{WindDir}_{\cos} = \cos\left(\theta_{\text{wind}}\right)$$

---

## 2. Regression Evaluation Metrics

Let $y_i$ denote the ground-truth active power and $\hat{y}_i$ denote the model prediction for sample $i \in \{1, \dots, N\}$.

### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

### Coefficient of Determination ($R^2$)
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}, \quad \text{where } \bar{y} = \frac{1}{N} \sum_{i=1}^N y_i$$

### Capacity-Normalized Errors (nMAE, nRMSE)
$$\text{nMAE} = \frac{\text{MAE}}{C_{\text{capacity}}} \times 100\%, \quad \text{nRMSE} = \frac{\text{RMSE}}{C_{\text{capacity}}} \times 100\%$$
Where $C_{\text{capacity}}$ is the confirmed rated nameplate capacity (e.g. 110 kW DC or AC equivalent).

---

## 3. Daylight Robust Anomaly Detection Metrics

### Median Absolute Deviation (MAD)
$$\text{MAD} = \text{median}\left(\left| r_i - \text{median}(r) \right|\right)$$
where residual $r_i = y_i - \hat{y}_i$ evaluated over daylit timestamps ($\text{GHI} > 50 \text{ W/m}^2$).

### Robust Standard Deviation Proxy
$$\sigma_{\text{robust}} = 1.4826 \cdot \text{MAD}$$

### Operational Anomaly Threshold
$$\text{Flag}_i = \begin{cases} 
1, & \text{if } |r_i| > 3 \cdot \sigma_{\text{robust}} \text{ and } \text{GHI}_i > 50 \text{ W/m}^2 \\
0, & \text{otherwise}
\end{cases}$$

