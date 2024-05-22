# Logistic Optimization Model

## Overview
This repository contains a logistic optimization model aimed at minimizing the total transportation cost while considering various constraints related to vehicle capacities, delivery times, and driver working hours.

## Objective Function
The objective of the model is to minimize the total transportation cost:

$$\text{Minimize:} \$$
$$Z = \sum_{k=1}^p \sum_{j=1}^n \sum_{i=1}^m (x_{i,j,k} \cdot c_{i,j}) \$$

Where:
- \( x_{i,j,k} \) — quantity of cargo \( j \) delivered to point \( i \) by driver \( k \).
- \( c_{i,j} \) — cost associated with transporting cargo \( j \) to point \( i \).

## Constraints
1. **Demand Satisfaction**:
   \[ \sum_{k=1}^p \sum_{j=1}^n (x_{i,j,k} \cdot d_i) \geq d_i \quad \forall i \]
   Ensure that the demand \( d_i \) at each point \( i \) is met.

2. **Cargo Quantity Limit**:
   \[ \sum_{k=1}^p \sum_{j=1}^n x_{i,j,k} \leq q_i \quad \forall i \]
   Ensure that the total quantity of cargo \( q_i \) does not exceed the capacity for each point \( i \).

3. **Delivery Time Limit**:
   \[ \sum_{k=1}^p \sum_{j=1}^n (x_{i,j,k} \cdot p_{i,j}) \leq 24 \quad \forall i \]
   Ensure that deliveries are made within a 24-hour period.

4. **Driver Work Hours Limit (Over Accounting Period)**:
   \[ \sum_{i=1}^m \sum_{j=1}^n t_{i,j,k} \leq T_{\text{max}} \quad \forall k \]
   Ensure that the total working hours of each driver \( k \) do not exceed the maximum allowed \( T_{\text{max}} \) over the accounting period (e.g., 40 hours per week).

5. **Daily Driving Time Limit**:
   \[ \text{drive\_time}_k \leq T_{\text{drive\_max}} \quad \forall k, \text{day} \]
   Ensure that the daily driving time does not exceed \( T_{\text{drive\_max}} \) (e.g., 9 hours per day, with an allowance of up to 10 hours twice a week).

6. **Weekly Driving Time Limit**:
   \[ \sum_{\text{day}} \text{drive\_time}_k \leq T_{\text{drive\_week}} \quad \forall k \]
   Ensure that the weekly driving time does not exceed \( T_{\text{drive\_week}} \) (e.g., 56 hours per week).

7. **Minimum Daily Rest**:
   \[ \text{rest\_time}_k \geq T_{\text{rest\_min}} \quad \forall k, \text{day} \]
   Ensure that each driver \( k \) gets a minimum daily rest period \( T_{\text{rest\_min}} \) (e.g., 11 hours).

8. **Special Breaks in Driving**:
   \[ \text{if} \ \text{drive\_time}_k \geq 4.5 \ \text{hours}, \ \text{then} \ \text{special break} \ \geq 45 \ \text{minutes} \]
   Ensure that drivers take special breaks after 4.5 hours of driving, unless they are taking a longer rest or meal break.

9. **Delivery Time Windows**:
   \[ a_i \leq t_{i,j,k} \leq b_i \quad \forall i, \forall j, \forall k \]
   Ensure that deliveries to each point \( i \) occur within the specified time window \([a_i, b_i]\).

## How to Use
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/logistic-optimization.git
