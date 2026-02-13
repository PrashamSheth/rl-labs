**Student Name:** Prasham Sheth
**Roll Number:** 26

---

# 1. Introduction

The goal of this lab is to implement a **Contextual Multi-Armed Bandit (CMAB)** system for recommending news articles to users. Unlike traditional bandits, contextual bandits use additional information (context) about users to make better decisions.

In this lab:

* Users are classified into 3 categories:

  * User_1
  * User_2
  * User_3

* Each user context has 4 possible news categories:

  * Entertainment
  * Education
  * Tech
  * Crime

Thus, the total number of arms is:

[
Total Arms = 3 \times 4 = 12
]

The system learns which article category gives the highest reward for each user type.

---

# 2. Data Preprocessing

## 2.1 Handling Missing Values

Missing values were removed using:

```python
news_df.dropna(inplace=True)
train_users.dropna(inplace=True)
test_users.dropna(inplace=True)
```

This ensures that the model receives clean data without invalid entries.

---

## 2.2 Encoding Categorical Features

Categorical columns such as:

* browser_version
* region_code
* subscriber

were converted into numeric form using LabelEncoder.

Example:

```python
le = LabelEncoder()
X_encoded[col] = le.fit_transform(X_encoded[col])
```

This allows machine learning models to process categorical information.

---

## 2.3 Target Encoding

The target column `label` was encoded as:

| User Label | Encoded Value |
| ---------- | ------------- |
| user_1     | 0             |
| user_2     | 1             |
| user_3     | 2             |

This represents the context for the contextual bandit.

---

## 2.4 Train-Validation Split

The dataset was split into:

* 80% training
* 20% validation

using stratified sampling:

```python
train_test_split(..., stratify=y_encoded)
```

This preserves the distribution of user classes.

---

# 3. User Classification (Context Detection)

A classifier was trained to predict user category from user features.

Two classifiers were tested:

## Logistic Regression

Used for context detection.

## Random Forest Classifier

Used for improved classification performance.

Validation Accuracy:

[
Accuracy = 88.89%
]

Classification Summary:

| Class  | Precision | Recall | F1-Score |
| ------ | --------- | ------ | -------- |
| user_1 | 0.84      | 0.99   | 0.91     |
| user_2 | 0.94      | 0.95   | 0.95     |
| user_3 | 0.00      | 0.00   | 0.00     |

The classifier performs well overall, but performance on user_3 is lower due to class imbalance.

This classifier provides the **context input** to the contextual bandit.

---

# 4. Contextual Bandit Formulation

Each arm represents:

[
Arm = (User Context, News Category)
]

Mapping:

| Arm Index | User Context | Category                              |
| --------- | ------------ | ------------------------------------- |
| 0-3       | User_1       | Entertainment, Education, Tech, Crime |
| 4-7       | User_2       | Entertainment, Education, Tech, Crime |
| 8-11      | User_3       | Entertainment, Education, Tech, Crime |

Reward sampler initialized using roll number:

```python
reward_sampler = sampler(26)
```

Rewards are obtained using:

```python
reward_sampler.sample(arm_index)
```

---

# 5. Bandit Algorithms Implemented

Three contextual bandit algorithms were implemented:

---

## 5.1 Epsilon-Greedy

Strategy:

* With probability ε → explore
* Otherwise → exploit best arm

Formula:

[
a =
\begin{cases}
random & \text{with probability } \epsilon \
argmax(Q) & \text{otherwise}
\end{cases}
]

ε values tested:

* 0.05
* 0.1
* 0.2

---

## 5.2 Upper Confidence Bound (UCB)

Formula:

[
UCB(a) = Q(a) + c \sqrt{\frac{\log t}{N(a)}}
]

This balances:

* exploitation
* exploration using uncertainty

Parameter used:

[
c = 2
]

---

## 5.3 SoftMax Strategy

Formula:

[
P(a) =
\frac{e^{Q(a)/\tau}}
{\sum e^{Q(i)/\tau}}
]

Temperature parameter:

[
\tau = 0.1
]

Lower temperature favors exploitation.

---

# 6. Reinforcement Learning Simulation

Simulation was run for:

[
5000 \text{ steps}
]

At each step:

1. Context predicted using classifier
2. Algorithm selects arm
3. Reward obtained
4. Estimates updated

Average reward was recorded over time.

---

# 7. Results and Analysis

## 7.1 Average Reward vs Time

Observed performance:

| Algorithm      | Final Avg Reward      |
| -------------- | --------------------- |
| UCB            | Highest (~10.6)       |
| Epsilon-Greedy | Slightly lower (~9.8) |
| SoftMax        | Lowest (~6.8)         |

Observations:

* UCB achieved highest performance
* Epsilon-Greedy performed well
* SoftMax had slower convergence

UCB balances exploration efficiently.

---

## 7.2 Epsilon Hyperparameter Comparison

Tested values:

| ε    | Final Reward    |
| ---- | --------------- |
| 0.05 | Highest (~10.2) |
| 0.1  | Medium (~9.6)   |
| 0.2  | Lowest (~8.8)   |

Observation:

Lower epsilon performs better.

Reason:

Less random exploration, more exploitation of optimal arms.

---

# 8. Observations

Key findings:

1. Context improves recommendation performance significantly.

2. UCB performs best because:

   * It balances exploration intelligently
   * It considers uncertainty

3. Epsilon-Greedy performs well but wastes exploration randomly.

4. SoftMax performs worst due to probabilistic selection.

5. Lower epsilon gives better performance after learning.

---

# 9. Comparison of Algorithms

| Algorithm      | Strengths                                 | Weaknesses                     |
| -------------- | ----------------------------------------- | ------------------------------ |
| Epsilon-Greedy | Simple, effective                         | Random exploration inefficient |
| UCB            | Best performance, intelligent exploration | Slightly complex               |
| SoftMax        | Smooth exploration                        | Slower convergence             |

Best algorithm:

[
\boxed{UCB}
]

---

# 10. Strengths of Contextual Bandit

* Uses user information
* Learns optimal recommendation
* Improves over time
* Adapts to different users

---

# 11. Limitations

* Requires good context classifier
* Needs sufficient training data
* Poor classifier reduces performance
* Class imbalance affects accuracy

---

# 12. Conclusion

In this lab, a contextual bandit system was successfully implemented to recommend news articles based on user context.

Key points:

* Built user classifier
* Implemented Epsilon-Greedy, UCB, SoftMax
* Simulated reinforcement learning
* Compared performance

Final Result:

[
\boxed{\text{UCB achieved the highest reward and best performance}}
]

Contextual bandits provide an effective method for personalized recommendations.

