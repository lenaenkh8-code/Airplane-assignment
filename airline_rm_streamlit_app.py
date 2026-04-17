from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Airline Revenue Management Simulator",
    page_icon="✈️",
    layout="wide",
)


@dataclass
class RMParams:
    seats: int = 150
    total_ticket_limit: int = 170
    early_booking_limit: int = 145
    price_early: float = 500.0
    price_late: float = 1500.0
    demand_early_mean: float = 200.0
    demand_early_sd: float = 20.0
    demand_late_mean: float = 20.0
    demand_late_sd: float = 5.0
    no_show_early: float = 0.05
    no_show_late: float = 0.15
    late_refund_rate: float = 1.0
    volunteer_prob_early: float = 0.015
    voucher_cost: float = 800.0
    involuntary_cost: float = 3000.0
    simulations: int = 25000
    seed: int = 42


def bounded_normal_int(rng: np.random.Generator, mean: float, sd: float, size: int) -> np.ndarray:
    draws = np.rint(rng.normal(mean, sd, size)).astype(int)
    return np.maximum(draws, 0)


@st.cache_data(show_spinner=False)
def run_simulation(params: RMParams) -> pd.DataFrame:
    rng = np.random.default_rng(params.seed)
    n = int(params.simulations)

    early_demand = bounded_normal_int(rng, params.demand_early_mean, params.demand_early_sd, n)
    late_demand = bounded_normal_int(rng, params.demand_late_mean, params.demand_late_sd, n)

    early_sold = np.minimum(early_demand, params.early_booking_limit)
    remaining_for_late = np.maximum(params.total_ticket_limit - early_sold, 0)
    late_sold = np.minimum(late_demand, remaining_for_late)

    early_show = rng.binomial(early_sold, 1 - params.no_show_early)
    late_show = rng.binomial(late_sold, 1 - params.no_show_late)
    total_show = early_show + late_show

    oversold = np.maximum(total_show - params.seats, 0)
    volunteers_available = rng.binomial(early_show, params.volunteer_prob_early)
    voluntary_denied = np.minimum(oversold, volunteers_available)
    involuntary_denied = np.maximum(oversold - voluntary_denied, 0)

    late_refund_cost = (late_sold - late_show) * params.price_late * params.late_refund_rate
    voluntary_cost = voluntary_denied * params.voucher_cost
    involuntary_cost = involuntary_denied * params.involuntary_cost

    revenue_early = early_sold * params.price_early
    revenue_late_gross = late_sold * params.price_late
    profit = revenue_early + revenue_late_gross - late_refund_cost - voluntary_cost - involuntary_cost

    boarded = np.minimum(total_show, params.seats)
    load_factor = boarded / params.seats
    empty_seats = np.maximum(params.seats - total_show, 0)

    return pd.DataFrame(
        {
            "profit": profit,
            "revenue_early": revenue_early,
            "revenue_late_gross": revenue_late_gross,
            "late_refund_cost": late_refund_cost,
            "voluntary_cost": voluntary_cost,
            "involuntary_cost": involuntary_cost,
            "early_sold": early_sold,
            "late_sold": late_sold,
            "early_show": early_show,
            "late_show": late_show,
            "total_show": total_show,
            "boarded": boarded,
            "empty_seats": empty_seats,
            "oversold": oversold,
            "voluntary_denied": voluntary_denied,
            "involuntary_denied": involuntary_denied,
            "load_factor": load_factor,
        }
    )


@st.cache_data(show_spinner=False)
def optimize_policy(base_params: RMParams, total_min: int, total_max: int, early_min: int, early_max: int) -> pd.DataFrame:
    records = []
    for total_limit in range(total_min, total_max + 1):
        for early_limit in range(early_min, min(early_max, total_limit) + 1):
            test_params = RMParams(**{**base_params.__dict__, "total_ticket_limit": total_limit, "early_booking_limit": early_limit})
            df = run_simulation(test_params)
            records.append(
                {
                    "total_ticket_limit": total_limit,
                    "early_booking_limit": early_limit,
                    "protected_for_late": total_limit - early_limit,
                    "expected_profit": df["profit"].mean(),
                    "p5_profit": df["profit"].quantile(0.05),
                    "oversold_rate": (df["oversold"] > 0).mean(),
                    "involuntary_db_rate": (df["involuntary_denied"] > 0).mean(),
                    "avg_load_factor": df["load_factor"].mean(),
                }
            )
    return pd.DataFrame(records).sort_values("expected_profit", ascending=False).reset_index(drop=True)



def fmt_money(value: float) -> str:
    return f"${value:,.0f}"



def fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


st.title("✈️ Airline Revenue Management Simulator")
st.caption("Defaults match the airline revenue management case assumptions.")

with st.sidebar:
    st.header("Inputs")
    st.subheader("Flight and pricing")
    seats = st.number_input("Available seats", min_value=1, value=150, step=1)
    total_ticket_limit = st.number_input("Total tickets to offer for sale", min_value=1, value=170, step=1)
    early_booking_limit = st.number_input(
        "Booking limit for F2 / leisure fare",
        min_value=0,
        max_value=int(total_ticket_limit),
        value=min(145, int(total_ticket_limit)),
        step=1,
        help="Maximum number of lower-fare tickets sold before the final 2 weeks.",
    )
    st.metric("Protected for F1 / business fare", max(int(total_ticket_limit) - int(early_booking_limit), 0))

    price_early = st.number_input("F2 / leisure fare price", min_value=0.0, value=500.0, step=50.0)
    price_late = st.number_input("F1 / business fare price", min_value=0.0, value=1500.0, step=50.0)

    st.subheader("Demand assumptions")
    demand_early_mean = st.number_input("F2 / leisure demand mean", min_value=0.0, value=200.0, step=1.0)
    demand_early_sd = st.number_input("F2 / leisure demand SD", min_value=0.0, value=20.0, step=1.0)
    demand_late_mean = st.number_input("F1 / business demand mean", min_value=0.0, value=20.0, step=1.0)
    demand_late_sd = st.number_input("F1 / business demand SD", min_value=0.0, value=5.0, step=1.0)

    st.subheader("No-show and denied boarding assumptions")
    no_show_early = st.slider("F2 / leisure no-show probability", 0.0, 0.5, 0.05, 0.005)
    no_show_late = st.slider("F1 / business no-show probability", 0.0, 0.5, 0.15, 0.005)
    late_refund_rate = st.slider("Refund share on F1 / business no-shows", 0.0, 1.0, 1.0, 0.05)
    volunteer_prob_early = st.slider("Volunteer probability among F2 show-ups", 0.0, 0.2, 0.015, 0.001)
    voucher_cost = st.number_input("Voucher cost per voluntary denied boarding", min_value=0.0, value=800.0, step=50.0)
    involuntary_cost = st.number_input("Cost per involuntary denied boarding", min_value=0.0, value=3000.0, step=100.0)

    st.subheader("Simulation settings")
    simulations = st.selectbox("Monte Carlo runs", [5000, 10000, 25000, 50000], index=2)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

params = RMParams(
    seats=int(seats),
    total_ticket_limit=int(total_ticket_limit),
    early_booking_limit=int(early_booking_limit),
    price_early=float(price_early),
    price_late=float(price_late),
    demand_early_mean=float(demand_early_mean),
    demand_early_sd=float(demand_early_sd),
    demand_late_mean=float(demand_late_mean),
    demand_late_sd=float(demand_late_sd),
    no_show_early=float(no_show_early),
    no_show_late=float(no_show_late),
    late_refund_rate=float(late_refund_rate),
    volunteer_prob_early=float(volunteer_prob_early),
    voucher_cost=float(voucher_cost),
    involuntary_cost=float(involuntary_cost),
    simulations=int(simulations),
    seed=int(seed),
)

sim_df = run_simulation(params)

expected_profit = sim_df["profit"].mean()
profit_p5 = sim_df["profit"].quantile(0.05)
profit_p95 = sim_df["profit"].quantile(0.95)
oversold_rate = (sim_df["oversold"] > 0).mean()
involuntary_rate = (sim_df["involuntary_denied"] > 0).mean()
avg_load = sim_df["load_factor"].mean()
avg_boarded = sim_df["boarded"].mean()
avg_empty = sim_df["empty_seats"].mean()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Expected profit", fmt_money(expected_profit))
m2.metric("5th percentile profit", fmt_money(profit_p5))
m3.metric("Average load factor", fmt_pct(avg_load))
m4.metric("Oversold scenario rate", fmt_pct(oversold_rate))
m5.metric("Involuntary DB scenario rate", fmt_pct(involuntary_rate))

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Optimization", "Policy comparison", "Method"])

with tab1:
    left, right = st.columns([0.95, 1.05])

    with left:
        st.subheader("Summary")
        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Expected profit",
                    "5th percentile profit",
                    "95th percentile profit",
                    "Average boarded passengers",
                    "Average empty seats",
                    "Average involuntary denied boardings",
                    "Protected seats for F1 / business",
                ],
                "Value": [
                    fmt_money(expected_profit),
                    fmt_money(profit_p5),
                    fmt_money(profit_p95),
                    f"{avg_boarded:.1f}",
                    f"{avg_empty:.1f}",
                    f"{sim_df['involuntary_denied'].mean():.3f}",
                    f"{params.total_ticket_limit - params.early_booking_limit}",
                ],
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        revenue_df = pd.DataFrame(
            {
                "Component": [
                    "F2 revenue",
                    "F1 revenue",
                    "F1 no-show refunds",
                    "Voluntary DB cost",
                    "Involuntary DB cost",
                ],
                "Average amount": [
                    sim_df["revenue_early"].mean(),
                    sim_df["revenue_late_gross"].mean(),
                    -sim_df["late_refund_cost"].mean(),
                    -sim_df["voluntary_cost"].mean(),
                    -sim_df["involuntary_cost"].mean(),
                ],
            }
        )
        fig_components = px.bar(revenue_df, x="Component", y="Average amount", text="Average amount")
        fig_components.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_components.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Average amount per simulation")
        st.plotly_chart(fig_components, use_container_width=True)

    with right:
        st.subheader("Profit distribution")
        fig_profit = px.histogram(sim_df, x="profit", nbins=40)
        fig_profit.add_vline(x=expected_profit, line_dash="dash", annotation_text="Mean")
        fig_profit.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Profit", yaxis_title="Frequency")
        st.plotly_chart(fig_profit, use_container_width=True)

        risk_df = pd.DataFrame(
            {
                "Outcome": ["Any oversold", "Any involuntary denied boarding"],
                "Probability": [oversold_rate, involuntary_rate],
            }
        )
        fig_risk = px.bar(risk_df, x="Outcome", y="Probability", text="Probability")
        fig_risk.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig_risk.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), yaxis_tickformat=".0%")
        st.plotly_chart(fig_risk, use_container_width=True)

with tab2:
    st.subheader("Search for the best policy")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total_min = st.number_input("Min total tickets", min_value=1, value=max(params.seats - 5, 1), step=1)
    with c2:
        total_max = st.number_input("Max total tickets", min_value=int(total_min), value=params.seats + 20, step=1)
    with c3:
        early_min = st.number_input("Min F2 booking limit", min_value=0, value=max(params.seats - 20, 0), step=1)
    with c4:
        early_max = st.number_input("Max F2 booking limit", min_value=int(early_min), value=params.seats, step=1)

    opt_df = optimize_policy(params, int(total_min), int(total_max), int(early_min), int(early_max))
    best = opt_df.iloc[0]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Best total tickets", int(best["total_ticket_limit"]))
    r2.metric("Best F2 booking limit", int(best["early_booking_limit"]))
    r3.metric("Protected for F1", int(best["protected_for_late"]))
    r4.metric("Best expected profit", fmt_money(best["expected_profit"]))

    st.info(
        f"Under the current assumptions, the best policy in this search is to sell up to {int(best['total_ticket_limit'])} total tickets and cap F2 / leisure sales at {int(best['early_booking_limit'])}."
    )

    heatmap = opt_df.pivot(index="early_booking_limit", columns="total_ticket_limit", values="expected_profit")
    fig_heat = px.imshow(
        heatmap.sort_index(ascending=True),
        aspect="auto",
        labels={"x": "Total tickets", "y": "F2 booking limit", "color": "Expected profit"},
    )
    fig_heat.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

    top_display = opt_df.head(10).copy()
    for col in ["expected_profit", "p5_profit"]:
        top_display[col] = top_display[col].map(fmt_money)
    for col in ["oversold_rate", "involuntary_db_rate", "avg_load_factor"]:
        top_display[col] = top_display[col].map(fmt_pct)
    st.dataframe(top_display, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Compare policies")
    default_policies = pd.DataFrame(
        [
            {"Label": "Case default", "Total tickets": 170, "F2 booking limit": 145},
            {"Label": "Capacity only", "Total tickets": params.seats, "F2 booking limit": min(params.seats, 130)},
            {"Label": "Optimized candidate", "Total tickets": max(params.seats + 10, params.total_ticket_limit), "F2 booking limit": min(max(params.seats - 10, 0), max(params.seats + 10, params.total_ticket_limit))},
        ]
    )
    edited = st.data_editor(default_policies, num_rows="fixed", use_container_width=True)

    records = []
    for _, row in edited.iterrows():
        total_limit = int(row["Total tickets"])
        early_limit = int(min(row["F2 booking limit"], total_limit))
        label = str(row["Label"])
        test_params = RMParams(**{**params.__dict__, "total_ticket_limit": total_limit, "early_booking_limit": early_limit})
        df = run_simulation(test_params)
        records.append(
            {
                "Policy": label,
                "Total tickets": total_limit,
                "F2 booking limit": early_limit,
                "Protected for F1": total_limit - early_limit,
                "Expected profit": df["profit"].mean(),
                "5th percentile profit": df["profit"].quantile(0.05),
                "Oversold rate": (df["oversold"] > 0).mean(),
                "Involuntary DB rate": (df["involuntary_denied"] > 0).mean(),
            }
        )

    compare_df = pd.DataFrame(records)
    display_compare = compare_df.copy()
    for col in ["Expected profit", "5th percentile profit"]:
        display_compare[col] = display_compare[col].map(fmt_money)
    for col in ["Oversold rate", "Involuntary DB rate"]:
        display_compare[col] = display_compare[col].map(fmt_pct)
    st.dataframe(display_compare, use_container_width=True, hide_index=True)

    pc1, pc2 = st.columns(2)
    with pc1:
        fig_compare_profit = px.bar(compare_df, x="Policy", y="Expected profit", text="Expected profit")
        fig_compare_profit.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_compare_profit.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_compare_profit, use_container_width=True)
    with pc2:
        melt = compare_df.melt(id_vars="Policy", value_vars=["Oversold rate", "Involuntary DB rate"], var_name="Metric", value_name="Rate")
        fig_compare_risk = px.bar(melt, x="Policy", y="Rate", color="Metric", barmode="group", text="Rate")
        fig_compare_risk.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig_compare_risk.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20), yaxis_tickformat=".0%")
        st.plotly_chart(fig_compare_risk, use_container_width=True)

with tab4:
    st.subheader("How the model works")
    st.markdown(
        """
1. F2 / leisure demand is realized first, up to the leisure booking limit.
2. F1 / business demand arrives later and uses the remaining inventory up to the total ticket limit.
3. Show-ups are simulated using class-specific no-show probabilities.
4. If show-ups exceed seat capacity, the model first tries voluntary denied boarding from F2 show-ups.
5. Any remaining excess passengers are treated as involuntary denied boardings.
6. Profit equals ticket revenue minus F1 no-show refunds and denied-boarding costs.
        """
    )

    assumptions = pd.DataFrame(
        {
            "Parameter": [
                "Seats",
                "F2 price",
                "F1 price",
                "F2 demand",
                "F1 demand",
                "F2 no-show probability",
                "F1 no-show probability",
                "Voucher cost",
                "Involuntary DB cost",
            ],
            "Current value": [
                params.seats,
                fmt_money(params.price_early),
                fmt_money(params.price_late),
                f"N({params.demand_early_mean:.0f}, {params.demand_early_sd:.0f})",
                f"N({params.demand_late_mean:.0f}, {params.demand_late_sd:.0f})",
                fmt_pct(params.no_show_early),
                fmt_pct(params.no_show_late),
                fmt_money(params.voucher_cost),
                fmt_money(params.involuntary_cost),
            ],
        }
    )
    st.dataframe(assumptions, use_container_width=True, hide_index=True)

st.caption("Built for interactive Monte Carlo testing of airline overbooking and fare protection policies.")
