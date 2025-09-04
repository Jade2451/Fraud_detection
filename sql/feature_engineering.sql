-- This query engineers new features from the raw transaction data.
-- Final SELECT statement that joins the engineered features back to the original data.
SELECT
    t.*,
    -- Feature 1: Deviation from the user's average transaction amount over the last 24 hours.
    -- A large positive deviation might indicate a fraudulent transaction.
    (t.Amount - fe.avg_amount_24h_user) AS amount_deviation_24h_user,

    -- Feature 2: Number of transactions by the user in the last 24 hours.
    -- A sudden spike in transactions could be a sign of fraud.
    fe.transaction_count_24h_user,

    -- Feature 3: Time (in seconds) since the user's last transaction.
    -- Very short intervals between transactions (e.g., from different locations) can be suspicious.
    fe.time_since_last_transaction_user
FROM
    transactions t
LEFT JOIN (
    -- This subquery calculates the features for each transaction using window functions.
    SELECT
        "index",
        -- Calculate the user's average transaction amount over a 24-hour window.
        -- The window is defined as all transactions by that user in the 86400 seconds (24 hours) preceding the current one.
        AVG(Amount) OVER (
            PARTITION BY user_id
            ORDER BY Time
            RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING
        ) AS avg_amount_24h_user,

        -- Count the number of transactions by the user in the same 24-hour window.
        COUNT("index") OVER (
            PARTITION BY user_id
            ORDER BY Time
            RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING
        ) AS transaction_count_24h_user,

        -- Calculate the time difference between the current and the previous transaction for the same user.
        -- LAG(Time, 1, Time) fetches the 'Time' from the previous row for the same user. If no previous row exists, it uses the current time, making the difference 0.
        (Time - LAG(Time, 1, Time) OVER (
            PARTITION BY user_id
            ORDER BY Time
        )) AS time_since_last_transaction_user
    FROM
        transactions
) AS fe ON t."index" = fe."index";