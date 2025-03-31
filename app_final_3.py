import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


meals_df = pd.read_csv("meals.csv")
meal_ids = meals_df["Meal_ID"].tolist()

# ----------------------------
# Fuzzy Logic Functions
# ----------------------------
def triangular(x, a, b, c):
    """Triangular membership function."""
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

def age_membership(age):
    return {
        "Young": triangular(age, 0, 18, 30),
        "Adult": triangular(age, 25, 40, 60),
        "Elder": triangular(age, 55, 70, 100)
    }

def bmi_membership(bmi):
    return {
        "Underweight": triangular(bmi, 0, 16, 18.5),
        "Normal": triangular(bmi, 18, 22, 25),
        "Overweight": triangular(bmi, 23, 27, 30),
        "Obese": triangular(bmi, 28, 35, 50)
    }

def activity_membership(activity_level):
    return {
        "Low": triangular(activity_level, 0, 2, 4),
        "Moderate": triangular(activity_level, 3, 5, 7),
        "High": triangular(activity_level, 6, 8, 10)
    }

def fuzzy_health_assessment(age, bmi, activity, diabetes=False, hypertension=False):
    """
    Compute a fuzzy risk score using weighted averages.
    For each factor, we assign:
      - Age: Young=0, Adult=40, Elder=80
      - BMI: Underweight=20, Normal=40, Overweight=60, Obese=80
      - Activity: Low=80, Moderate=50, High=20
    The final risk is the average, adjusted for diabetes (+5) and hypertension (+3).
    """
    age_m = age_membership(age)
    bmi_m = bmi_membership(bmi)
    act_m = activity_membership(activity)
    
    age_risk = {"Young": 0, "Adult": 40, "Elder": 80}
    bmi_risk = {"Underweight": 20, "Normal": 40, "Overweight": 60, "Obese": 80}
    act_risk = {"Low": 80, "Moderate": 50, "High": 20}
    
    age_risk_val = sum(age_m[cat] * age_risk[cat] for cat in age_m) / (sum(age_m.values()) + 1e-6)
    bmi_risk_val = sum(bmi_m[cat] * bmi_risk[cat] for cat in bmi_m) / (sum(bmi_m.values()) + 1e-6)
    act_risk_val = sum(act_m[cat] * act_risk[cat] for cat in act_m) / (sum(act_m.values()) + 1e-6)
    
    risk_score = (age_risk_val + bmi_risk_val + act_risk_val) / 3
    if diabetes:
        risk_score += 5
    if hypertension:
        risk_score += 3
    risk_score = np.clip(risk_score, 0, 100)
    
    if risk_score < 35:
        risk_category = "Low"
    elif risk_score < 60:
        risk_category = "Medium"
    else:
        risk_category = "High"
    
    explanation = {
        "Age Risk Value": age_risk_val,
        "BMI Risk Value": bmi_risk_val,
        "Activity Risk Value": act_risk_val,
        "Final Risk Score": risk_score
    }
    return risk_score, risk_category, explanation

def get_recommendations(age, risk_category):
    """
    Provide personalized nutrient recommendations based on risk and age.
    """
    if risk_category == "High":
        base_cal = 1800
        protein_rec = "High"
    elif risk_category == "Medium":
        base_cal = 2100
        protein_rec = "Moderate"
    else:
        base_cal = 2500
        protein_rec = "Moderate"
    if age > 60:
        base_cal -= 200
    return {
        "Recommended Calories": base_cal,
        "Protein": protein_rec,
        "Fat": "Low" if risk_category == "High" else "Moderate",
        "Carbs": "Moderate" if risk_category != "Low" else "High"
    }

# ----------------------------
# Genetic Algorithm (GA) Functions
# ----------------------------
POPULATION_SIZE = 20
NUM_DAYS = 7
MEALS_PER_DAY = 3
CHROMOSOME_LENGTH = NUM_DAYS * MEALS_PER_DAY
GENERATIONS = 50
TARGET_CALORIES = 2100

w_macro = 0.5
w_variety = 0.3
w_allergy = 0.2

def fitness_function(chromosome, vegetarian, no_dairy, high_protein):
    plan = meals_df[meals_df["Meal_ID"].isin(chromosome)]
    total_calories = plan["Calories"].sum()
    target_weekly = TARGET_CALORIES * 7
    macro_diff_score = abs(target_weekly - total_calories)
    variety_score = CHROMOSOME_LENGTH - len(set(chromosome))
    
    penalty = 0
    if vegetarian:
        non_veg = plan[plan["Vegetarian"]=="No"].shape[0]
        penalty += non_veg * 50
    if no_dairy:
        dairy = plan[plan["Contains_DAIRY"]=="Yes"].shape[0]
        penalty += dairy * 50
    protein_total = plan["Protein_g"].sum()
    avg_protein = protein_total / CHROMOSOME_LENGTH
    if high_protein:
        if avg_protein < 10:
            penalty += (10 - avg_protein) * 100
        else:
            penalty -= 50
    return w_macro * macro_diff_score + w_variety * variety_score + w_allergy * penalty

def create_chromosome():
    return [random.choice(meal_ids) for _ in range(CHROMOSOME_LENGTH)]

def tournament_selection(population, k=3, vegetarian=False, no_dairy=False, high_protein=False):
    selected = random.sample(population, k)
    selected.sort(key=lambda chromo: fitness_function(chromo, vegetarian, no_dairy, high_protein))
    return selected[0]

def crossover(parent1, parent2):
    day = random.randint(0, NUM_DAYS - 1)
    start = day * MEALS_PER_DAY
    end = start + MEALS_PER_DAY
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[start:end], child2[start:end] = parent2[start:end], parent1[start:end]
    return child1, child2

def mutate(chromosome, mutation_rate=0.15):
    new_chromosome = chromosome.copy()
    for i in range(CHROMOSOME_LENGTH):
        if random.random() < mutation_rate:
            new_chromosome[i] = random.choice(meal_ids)
    return new_chromosome

# Use session state to store GA results so that changing the selectbox doesn't rerun the GA.
if "ga_result" not in st.session_state:
    st.session_state.ga_result = None
    st.session_state.fitness_history = None

def run_ga(vegetarian, no_dairy, high_protein):
    population = [create_chromosome() for _ in range(POPULATION_SIZE)]
    best_chromo = None
    best_fit = float('inf')
    fitness_history = []
    for generation in range(GENERATIONS):
        new_population = []
        population.sort(key=lambda chromo: fitness_function(chromo, vegetarian, no_dairy, high_protein))
        new_population.extend(population[:2])
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, vegetarian=vegetarian, no_dairy=no_dairy, high_protein=high_protein)
            parent2 = tournament_selection(population, vegetarian=vegetarian, no_dairy=no_dairy, high_protein=high_protein)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate=0.15)
            child2 = mutate(child2, mutation_rate=0.15)
            new_population.extend([child1, child2])
        population = new_population[:POPULATION_SIZE]
        current_best = min(population, key=lambda chromo: fitness_function(chromo, vegetarian, no_dairy, high_protein))
        current_fit = fitness_function(current_best, vegetarian, no_dairy, high_protein)
        if current_fit < best_fit:
            best_fit = current_fit
            best_chromo = current_best
        fitness_history.append(best_fit)
    return best_chromo, fitness_history

# ----------------------------
# Streamlit App Layout with Tabs
# ----------------------------
st.set_page_config(page_title="Fuzzy-GA Diet Planner", layout="wide")
st.title("Fuzzy-GA Powered Diet Planner")

with st.sidebar:
    st.title("Fuzzy-GA Diet Planner")
    st.markdown(
        """
        **About:**  
        This app uses fuzzy logic to assess your health risk based on your Age, BMI, and Activity Level.
        It then provides personalized nutrient recommendations and uses a genetic algorithm (GA)
        to generate an optimized weekly meal plan tailored to your dietary preferences.
        """
    )

# Tabs for Navigation
tabs = st.tabs(["Overview", "Health Assessment", "Meal Plan Optimization", "Membership Functions"])

# ----------------------------
# Tab 1: Overview
# ----------------------------
with tabs[0]:
    st.header("Overview")
    st.markdown(
        """
        **Fuzzy-GA Powered Diet Planner** integrates:
        - A **Fuzzy Logic Engine** to evaluate your health risk and provide nutrient recommendations.
        - A **Genetic Algorithm** to optimize a weekly meal plan that fits your nutritional needs and dietary preferences.
        
        Use the tabs to explore:
        - Your health assessment based on your inputs.
        - The optimized meal plan and detailed nutrient breakdowns.
        - Interactive membership function plots that explain the fuzzy logic.
        """
    )

# ----------------------------
# Tab 2: Health Assessment
# ----------------------------
with tabs[1]:
    st.header("Health Assessment")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Enter Your Health Details")
        user_age = st.number_input("Age", min_value=0, max_value=120, value=30, key="age")
        user_bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0, step=0.1, key="bmi")
        user_activity = st.slider("Activity Level (0 low - 10 high)", 0, 10, 5, key="activity")
        user_diabetes = st.checkbox("Diabetes", value=False, key="diabetes")
        user_hypertension = st.checkbox("Hypertension", value=False, key="hypertension")
    with col2:
        risk, risk_cat, fuzzy_expl = fuzzy_health_assessment(user_age, user_bmi, user_activity, user_diabetes, user_hypertension)
        recs = get_recommendations(user_age, risk_cat)
        st.subheader("Your Health Summary")
        st.markdown(f"**Risk Score:** {risk:.2f} ({risk_cat} Risk)")
        st.markdown("**Nutrient Recommendations:**")
        st.write(recs)
    st.markdown("---")
    st.markdown("**Fuzzy Logic Explanation:**")
    st.json(fuzzy_expl)

# ----------------------------
# Tab 3: Meal Plan Optimization
# ----------------------------
with tabs[2]:
    st.header("Meal Plan Optimization")
    st.markdown("Click the button below to generate an optimized weekly meal plan based on your dietary preferences.")
    dietary_veg = st.checkbox("Vegetarian", value=False, key="veg")
    dietary_no_dairy = st.checkbox("No Dairy", value=False, key="no_dairy")
    dietary_high_protein = st.checkbox("High Protein", value=False, key="high_protein")
    
    # Check if GA result is stored in session_state; if not, run GA when button clicked.
    if st.button("Generate Meal Plan"):
        with st.spinner("Optimizing your meal plan..."):
            best_plan, fitness_history = run_ga(dietary_veg, dietary_no_dairy, dietary_high_protein)
            st.session_state.ga_result = best_plan
            st.session_state.fitness_history = fitness_history
        st.success("Meal plan generated!")
    
    if st.session_state.ga_result is not None:
        # GA Evolution Chart
        st.subheader("GA Evolution")
        fig_ga, ax_ga = plt.subplots(figsize=(8, 4))
        ax_ga.plot(st.session_state.fitness_history, marker='o', linestyle='-')
        ax_ga.set_xlabel("Generation")
        ax_ga.set_ylabel("Best Fitness")
        ax_ga.set_title("GA Evolution: Fitness over Generations")
        st.pyplot(fig_ga)
        
        # Display Optimized Weekly Meal Plan
        plan_df = pd.DataFrame(np.array(st.session_state.ga_result).reshape(NUM_DAYS, MEALS_PER_DAY),
                               columns=["Meal 1", "Meal 2", "Meal 3"])
        st.subheader("Optimized Weekly Meal Plan")
        st.dataframe(plan_df)
        
        # Daily Nutrient Totals
        daily_totals = []
        for day in range(NUM_DAYS):
            day_ids = st.session_state.ga_result[day*MEALS_PER_DAY:(day+1)*MEALS_PER_DAY]
            day_plan = meals_df[meals_df["Meal_ID"].isin(day_ids)]
            totals = {
                "Calories": day_plan["Calories"].sum(),
                "Protein": day_plan["Protein_g"].sum(),
                "Carbohydrates": day_plan["Carbohydrate_g"].sum(),
                "Fat": day_plan["Fat_g"].sum(),
                "Fiber": day_plan["Fiber_g"].sum()
            }
            daily_totals.append(totals)
        daily_df = pd.DataFrame(daily_totals)
        st.subheader("Daily Nutrient Totals")
        st.dataframe(daily_df)
        
        # Interactive Meal Nutrient Breakdown
        st.subheader("View Meal Nutrient Breakdown")
        # Use a selectbox with a key to preserve state
        meal_choice = st.selectbox("Select a Meal ID from the Optimized Plan", sorted(set(st.session_state.ga_result)), key="meal_select")
        selected_meal = meals_df[meals_df["Meal_ID"]==meal_choice].iloc[0]
        fig_meal, ax_meal = plt.subplots(figsize=(8, 4))
        meal_nutrients = ["Calories", "Protein_g", "Carbohydrate_g", "Fat_g", "Fiber_g"]
        nutrient_values = [selected_meal[n] for n in meal_nutrients]
        ax_meal.bar(meal_nutrients, nutrient_values, color="skyblue")
        ax_meal.set_ylabel("Value")
        ax_meal.set_title(f"Nutrient Breakdown: {selected_meal['Meal_Name']}")
        st.pyplot(fig_meal)
    else:
        st.info("Click 'Generate Meal Plan' to optimize your weekly meal plan.")

# ----------------------------
# Tab 4: Membership Functions
# ----------------------------
with tabs[3]:
    st.header("Membership Function Plots")
    st.markdown("These plots illustrate the membership functions used for Age, BMI, and Activity.")
    
    st.subheader("Age Membership Functions")
    x_age = np.linspace(0, 100, 500)
    fig_age, ax_age = plt.subplots(figsize=(8, 4))
    ax_age.plot(x_age, [age_membership(x)["Young"] for x in x_age], label="Young")
    ax_age.plot(x_age, [age_membership(x)["Adult"] for x in x_age], label="Adult")
    ax_age.plot(x_age, [age_membership(x)["Elder"] for x in x_age], label="Elder")
    ax_age.set_xlabel("Age")
    ax_age.set_ylabel("Membership Degree")
    ax_age.set_title("Age Membership Functions")
    ax_age.legend()
    st.pyplot(fig_age)
    
    st.subheader("BMI Membership Functions")
    x_bmi = np.linspace(0, 50, 500)
    fig_bmi, ax_bmi = plt.subplots(figsize=(8, 4))
    ax_bmi.plot(x_bmi, [bmi_membership(x)["Underweight"] for x in x_bmi], label="Underweight")
    ax_bmi.plot(x_bmi, [bmi_membership(x)["Normal"] for x in x_bmi], label="Normal")
    ax_bmi.plot(x_bmi, [bmi_membership(x)["Overweight"] for x in x_bmi], label="Overweight")
    ax_bmi.plot(x_bmi, [bmi_membership(x)["Obese"] for x in x_bmi], label="Obese")
    ax_bmi.set_xlabel("BMI")
    ax_bmi.set_ylabel("Membership Degree")
    ax_bmi.set_title("BMI Membership Functions")
    ax_bmi.legend()
    st.pyplot(fig_bmi)
    
    st.subheader("Activity Membership Functions")
    x_act = np.linspace(0, 10, 500)
    fig_act, ax_act = plt.subplots(figsize=(8, 4))
    ax_act.plot(x_act, [activity_membership(x)["Low"] for x in x_act], label="Low")
    ax_act.plot(x_act, [activity_membership(x)["Moderate"] for x in x_act], label="Moderate")
    ax_act.plot(x_act, [activity_membership(x)["High"] for x in x_act], label="High")
    ax_act.set_xlabel("Activity Level")
    ax_act.set_ylabel("Membership Degree")
    ax_act.set_title("Activity Membership Functions")
    ax_act.legend()
    st.pyplot(fig_act)
