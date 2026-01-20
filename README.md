# Music Recommendation & Statistical Analysis System

A comprehensive Python project that implements a **Collaborative Filtering Recommender System** and performs rigorous statistical analysis on user listening data. This project combines machine learning techniques with hypothesis testing to analyze track popularity and predict user preferences.

## Overview

This system is designed to analyze music data from two perspectives: **Statistical Inference** and **Predictive Modeling**. It uses real-world rating data to calculate confidence intervals, test hypotheses about song popularity, and build a recommendation engine that suggests tracks based on user similarity.

### Core Components:
* **Recommender Engine:** User-User Collaborative Filtering using Cosine Similarity.
* **Statistical Analysis:** Z-tests, T-tests, and Confidence Interval calculations.
* **Hypothesis Testing:** A/B testing to compare user preference between different tracks.
* **Evaluation:** Model performance tracking using Mean Squared Error (MSE).

## Project Structure

The project is modularized into specific analytical tasks:

| Module | Description |
| :--- | :--- |
| **`recommender.py`** | The core engine. Implements User-User Collaborative Filtering to predict ratings and recommend top tracks. |
| **`part1.py`** | **Confidence Intervals:** Calculates 95% confidence intervals for the average ratings of specific tracks using both Z-distribution and T-distribution. |
| **`part2.py`** | **Hypothesis Testing:** Performs two-sample t-tests to determine if the difference in ratings between two tracks is statistically significant (rejecting or failing to reject $H_0$). |
| **`part3.py`** | **Probability Analysis:** Calculates the probability of specific rating sequences and predicts the next rating in a chain based on prior probabilities. |
| **`part4.py`** | **Model Evaluation:** Splits data into Training/Test sets (80/20), trains the recommender, and calculates the Mean Squared Error (MSE) to measure accuracy. |

## Technologies

* **Language:** Python 3.x
* **Libraries:**
    * `pandas`: Data manipulation and CSV processing.
    * `numpy`: Matrix operations and numerical calculations.
    * `scipy.stats`: Statistical functions (T-tests, Norm/T-intervals).

## How to Run

### Prerequisites
Install the required libraries:
```bash
pip install pandas numpy scipy
```

### Running the Modules

You can run each part independently to see the analysis results:

1.  **Run the Statistical Analysis (Confidence Intervals):**
    ```bash
    python part1.py
    ```

2.  **Run the Hypothesis Testing (A/B Test):**
    ```bash
    python part2.py
    ```

3.  **Run the Probability/Sequence Analysis:**
    ```bash
    python part3.py
    ```

4.  **Train and Evaluate the Recommender (MSE Score):**
    ```bash
    python part4.py
    ```

## Methodology

### 1. Collaborative Filtering (`recommender.py`)
* Constructs a **User-Item Matrix** from `user_ratings.csv`.
* Calculates a **Cosine Similarity Matrix** to find users with similar tastes.
* Predicts a rating $P_{u,i}$ for user $u$ and item $i$ using the weighted average of neighbors' ratings.

### 2. Hypothesis Testing (`part2.py`)
* **Null Hypothesis ($H_0$):** The average rating of Track A is equal to Track B.
* **Alternative Hypothesis ($H_1$):** The average ratings are different.
* **Method:** Two-sample t-test assuming unequal variance (Welch's t-test).
* **Result:** Outputs p-values to determine significance at $\alpha = 0.05$.

### 3. Evaluation (`part4.py`)
* **Metric:** Mean Squared Error (MSE).
* **Process:** Hides 20% of ratings (Test Set), predicts them using the Training Set, and compares the predicted values vs. actual values.

## Data Format

The system expects CSV inputs in the following structure:
* **`user_ratings.csv`**: `[User ID, Track ID, Rating]`
* **`tracks.csv`**: `[Track ID, Track Name, Artist]`
