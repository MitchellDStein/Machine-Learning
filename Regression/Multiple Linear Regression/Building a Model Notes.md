# **Building a Model**

## **5 methods of building a model**

1. [All-In](#%22all-in%22---cases)
2. [Backwards Elimination](#backward-elimination)
3. [Forward Selection](#forward-selection)
4. [Bidirectional Elimination](#bidirectional-elimination)
5. Score Comparison

### Stepwise Recution

- [Backwards Elimination](#backward-elimination)
- [Forward Selection](#forward-selection)
- [Bidirectional Elimination](#bidirectional-elimination)

----------

## **"All-In" - cases:**

- Prior knowledge of variables
- You have to (following framework/orders)
- Preparing for Backwards Eliminiation

## **Backward Elimination:**

1. Select a significance level to stay in the model. (e.g. SL = 0.05)
2. Fit the full model with all possible predictors
3. Consider the predictor with the **highest** P-value.
    - If P > SL, go to step 4, otherwise go to ***FIN*** $\downarrow$
4. Remove the predictor (variable with the highest P-value)
5. Fit model without the variable*
    - Repeat to step 3

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;***FIN:*** Your model is ready

## **Forward Selection:**

1. Select a significance level to enter the model (e.g. SL = 0.05)
2. Fit all simple regression models $y \sim x_n$ Select the one with the lowest P-Value
3. Keep this variable and fit all possible models with one extra predictor added to one(s) you already have
4. Consider the predictor with the **lowest** P-value.
   - If P < SL, go to STEP 3

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\downarrow$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;***FIN:*** keep previous model

## **Bidirectional Elimination**

1. Select a significance level to enter and to stay in the model.
    - e.g. SL_ENTER = 0.05, SL_STAY = 0.05
2. Perform the next step of Forward Selection (new variables must have: P < SL_ENTER to enter)
3. Perform ALL steps of Backwards Elimination (old variables must have P < SL_STAY to stay)
   - Repeat to step 2
4. No new variables can enter and no old variables can exit

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\downarrow$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;***FIN:*** Your model is ready

## **All Possible Models**

1. Select a criterion of goodness of fit (e.g. Akaike criterion)
2. Construct All Possible Regression Models: $2^N-1$ total combinations
3. Select the one with the best criterion

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\downarrow$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;***FIN:*** Your model is ready
