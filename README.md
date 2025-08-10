## CARP: Customer Article Rebuy Prediction

This project implements a practical CARP (Customer Article Rebuy Prediction) pipeline inspired by Picnic's approach.

Key ideas:
- Focus on repeat purchases using handcrafted time-aware features
- Binary target: whether a customer will rebuy an article in their next order
- Time-aware training/validation split
- XGBoost classifier for tabular features

### Project layout

```
picinic_rec/
  data/
    raw/transactions.csv          # input data (sample provided)
    processed/carp_dataset.parquet # engineered dataset with features + labels
    predictions/recommendations.parquet
  models/
    carp_xgb.json                  # trained model
    feature_columns.json           # feature list used during training
  src/carp/
    __init__.py
    config.py
    data_loader.py
    feature_engineering.py
    labeling.py
    model.py
    serve.py
    utils.py
  main.py                          # simple CLI (build-dataset/train/predict)
  app.py                           # Streamlit UI
  requirements.txt
  README.md
```

### Data format

Input transactions should contain at least:

| customer_id | article_id | order_date | quantity |
|-------------|------------|------------|----------|
| C1          | A101       | 2024-05-01 | 1        |

Notes:
- `order_date` should be parseable to date (YYYY-MM-DD recommended)
- Multiple rows per (customer_id, order_date) are allowed (one per article)

### Install

```
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

### Quickstart

1) Build the training dataset (features + labels):

```
python main.py build-dataset \
  --input data/raw/transactions.csv \
  --output data/processed/carp_dataset.parquet
```

2) Train with a time-aware split (choose a cut-off date for validation):

```
python main.py train \
  --dataset data/processed/carp_dataset.parquet \
  --train-end 2024-05-31 \
  --model-out models/carp_xgb.json \
  --features-out models/feature_columns.json
```

3) Generate top-N recommendations per active customer (uses last order as reference time):

### Streamlit app

Run the UI (alternatively to CLI):

```
streamlit run app.py
```

In the left sidebar, choose:
- Build dataset: upload CSV or use default, then write parquet
- Train model: select dataset and date split, save model and features JSON
- Predict: upload CSV or use default, select model + features JSON, and write recommendations

```
python main.py predict \
  --input data/raw/transactions.csv \
  --model models/carp_xgb.json \
  --features models/feature_columns.json \
  --topn 5 \
  --output data/predictions/recommendations.parquet
```

### Feature overview (time-aware)

- Customer features: total orders so far, average days between orders, average basket size, number of unique articles, preferred day-of-week
- Article features: cumulative popularity (unique customers up to date), approximate repurchase cycle, peak month
- Interaction features: times customer bought article, days since last purchase, average gap between purchases, fraction of orders containing the article

Label: for each (customer, article) candidate at a reference order time t, label = 1 if the article appears in the next order, else 0.

### Tips

- Use realistic date cut-offs for training/validation to avoid leakage
- Refresh model weekly with rolling re-training
- For scale, consider persisting feature computations and using a feature store (e.g., Feast)


Here's a comprehensive list of all the features generated with detailed explanations:

## **Order-Level Features** (per customer, per order date)

### **Basket Features**
- **`basket_size`**: Total quantity of items purchased in the order
- **`num_lines`**: Number of unique articles in the order

### **Customer Order History Features**
- **`order_index`**: Sequential order number for each customer (1st, 2nd, 3rd order, etc.)
- **`prev_order_date`**: Date of the customer's previous order
- **`days_since_prev_order`**: Number of days between current and previous order
- **`avg_days_between_orders`**: Rolling average of days between orders for each customer
- **`cum_orders`**: Total number of orders made by the customer so far
- **`avg_basket_size`**: Rolling average basket size across all previous orders

### **Customer Article Diversity Features**
- **`unique_articles_so_far`**: Cumulative count of unique articles the customer has purchased up to the current order date

### **Day of Week Preference Features**
- **`dow_0` through `dow_6`**: One-hot encoded day of week (0=Monday, 6=Sunday)
- **`preferred_dow`**: The day of week the customer most frequently orders on (based on cumulative counts)

## **Article-Level Features** (per article, per order date)

### **Popularity Features**
- **`article_popularity`**: Cumulative count of unique customers who have purchased this article up to the current date

### **Article Behavior Features**
- **`article_median_repurchase_days`**: Median number of days between repeat purchases of the same article by the same customer
- **`article_peak_month`**: Month (1-12) when the article is most frequently purchased (seasonal pattern)

## **Customer-Article Interaction Features** (per customer, per article, per order date)

### **Purchase History Features**
- **`times_bought_so_far`**: Number of times the customer has purchased this specific article up to the current order
- **`last_purchase_date`**: Date when the customer last purchased this article
- **`days_since_last_purchase`**: Number of days since the customer's last purchase of this article

### **Purchase Pattern Features**
- **`avg_gap_days`**: Rolling average of days between purchases of this article by this customer
- **`fraction_orders_with_article`**: Fraction of the customer's orders that contained this article (up to current date)

## **Feature Assembly Process**

The final feature matrix is built by:

1. **Creating candidates**: All possible (customer, article, order_date) combinations where the customer has purchased the article on or before the order date
2. **Merging customer features**: Using direct joins for order-level features
3. **Merging interaction features**: Using as-of merges to get the most recent interaction data before each order date
4. **Merging article features**: Using as-of merges to get the most recent article popularity data before each order date

## **Key Technical Details**

- **As-of merges**: Used to ensure temporal consistency - features are only based on data available at or before the prediction time
- **Rolling calculations**: Many features use expanding windows to compute cumulative statistics
- **Robust merging**: The code includes fallback strategies for complex merge operations to handle edge cases
- **Temporal alignment**: All features are properly aligned to avoid data leakage from future information

This feature set captures customer behavior patterns, article popularity trends, and the temporal dynamics of purchasing behavior, making it suitable for a recommendation system that predicts which articles a customer is likely to purchase on a given order date.
