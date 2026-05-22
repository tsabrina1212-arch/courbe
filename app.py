import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(page_title="Courbes Mathématiques", page_icon="📈", layout="wide")

st.title("📈 Simulateur de Courbes Mathématiques")
st.caption("CPGE — Fonctions disponibles : sin, cos, tan, exp, log, sqrt, abs, pi, e, sinh, cosh, tanh, arcsin, arccos, arctan, ...")

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]

SAFE_NS = {k: getattr(np, k) for k in dir(np) if not k.startswith("_")}
SAFE_NS["pi"] = np.pi
SAFE_NS["e"] = np.e


def safe_eval(expr, namespace):
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = eval(expr, {"__builtins__": __builtins__}, namespace)
        arr = np.asarray(result, dtype=float)
        arr[~np.isfinite(arr)] = np.nan  # log(0), 1/0 → NaN (matplotlib saute ces points)
        return arr, None
    except Exception as ex:
        return None, str(ex)


tab_cart, tab_param, tab_polar = st.tabs(["Cartésien  y = f(x)", "Paramétrique  (x(t), y(t))", "Polaire  r(θ)"])


# ─── Onglet Cartésien ────────────────────────────────────────────────────────
with tab_cart:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Paramètres")

        if "curves" not in st.session_state:
            st.session_state.curves = [
            {"expr": "sin(x)", "label": "sin(x)", "color": COLORS[0]},
            {"expr": "1/x",    "label": "1/x",    "color": COLORS[1]},
        ]

        for i, curve in enumerate(st.session_state.curves):
            with st.expander(f"Courbe {i + 1}", expanded=True):
                curve["expr"] = st.text_input("Expression", value=curve["expr"], key=f"expr_{i}")
                curve["label"] = st.text_input("Légende", value=curve["label"], key=f"label_{i}")
                curve["color"] = st.color_picker("Couleur", value=curve["color"], key=f"color_{i}")
                if len(st.session_state.curves) > 1:
                    if st.button("Supprimer", key=f"del_{i}"):
                        st.session_state.curves.pop(i)
                        st.rerun()

        if len(st.session_state.curves) < 7:
            if st.button("➕ Ajouter une courbe"):
                n = len(st.session_state.curves)
                st.session_state.curves.append({
                    "expr": "cos(x)",
                    "label": f"g{n}(x)",
                    "color": COLORS[n % len(COLORS)],
                })
                st.rerun()

        st.divider()
        x_min, x_max = st.slider("Plage de x", -20.0, 20.0, (-2 * np.pi, 2 * np.pi), step=0.1)
        n_points = st.slider("Résolution (points)", 200, 5000, 1000, step=100)

        st.divider()
        st.subheader("Options")
        show_grid = st.checkbox("Grille", value=True)
        show_axes = st.checkbox("Axes centrés", value=True)
        show_deriv = st.checkbox("Afficher f'(x) (courbe 1)", value=False)
        show_tangent = st.checkbox("Tangente en x₀ (courbe 1)", value=False)
        if show_tangent:
            x0 = st.number_input("x₀", value=0.0, step=0.1)
        title = st.text_input("Titre du graphe", value="Courbes")
        y_auto = st.checkbox("Échelle Y automatique", value=True)
        if not y_auto:
            y_min, y_max = st.slider("Plage de y", -50.0, 50.0, (-5.0, 5.0), step=0.5)

    with col_right:
        x = np.linspace(x_min, x_max, n_points)
        fig, ax = plt.subplots(figsize=(10, 6))

        for curve in st.session_state.curves:
            ns = dict(SAFE_NS)
            ns["x"] = x
            y, err = safe_eval(curve["expr"], ns)
            if err:
                st.error(f"Erreur dans « {curve['expr']} » : {err}")
                continue
            ax.plot(x, y, color=curve["color"], label=curve["label"], linewidth=2)

        # Dérivée de la première courbe
        if show_deriv and st.session_state.curves:
            ns = dict(SAFE_NS)
            ns["x"] = x
            y0, err = safe_eval(st.session_state.curves[0]["expr"], ns)
            if y0 is not None:
                dy = np.gradient(y0, x)
                ax.plot(x, dy, "--", color="gray", label="f'(x)", linewidth=1.5, alpha=0.8)

        # Tangente en x₀
        if show_tangent and st.session_state.curves:
            ns = dict(SAFE_NS)
            ns["x"] = x
            y0, err = safe_eval(st.session_state.curves[0]["expr"], ns)
            if y0 is not None:
                dy = np.gradient(y0, x)
                idx = np.argmin(np.abs(x - x0))
                fx0 = y0[idx]
                fpx0 = dy[idx]
                tangent = fx0 + fpx0 * (x - x0)
                ax.plot(x, tangent, ":", color="orange", label=f"Tangente en x={x0:.2f}", linewidth=1.5)
                ax.scatter([x0], [fx0], color="orange", zorder=5)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if show_grid:
            ax.grid(True, alpha=0.3)
        if show_axes:
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axvline(0, color="black", linewidth=0.8)
        if not y_auto:
            ax.set_ylim(y_min, y_max)
        ax.legend()
        ax.set_xlim(x_min, x_max)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ─── Onglet Paramétrique ─────────────────────────────────────────────────────
with tab_param:
    col_left2, col_right2 = st.columns([1, 2])

    with col_left2:
        st.subheader("Paramètres")
        expr_x = st.text_input("x(t)", value="cos(t)", key="param_x")
        expr_y = st.text_input("y(t)", value="sin(t)", key="param_y")
        t_min, t_max = st.slider("Plage de t", -4 * np.pi, 4 * np.pi, (0.0, 2 * np.pi), step=0.1, key="t_range")
        n_pts_p = st.slider("Résolution", 200, 5000, 1000, step=100, key="n_pts_p")
        color_p = st.color_picker("Couleur", value="#d62728", key="color_p")
        label_p = st.text_input("Légende", value="(x(t), y(t))", key="label_p")
        title_p = st.text_input("Titre", value="Courbe paramétrique", key="title_p")
        grid_p = st.checkbox("Grille", value=True, key="grid_p")

    with col_right2:
        t = np.linspace(t_min, t_max, n_pts_p)
        ns_t = dict(SAFE_NS)
        ns_t["t"] = t

        xv, err_x = safe_eval(expr_x, ns_t)
        yv, err_y = safe_eval(expr_y, ns_t)

        if err_x:
            st.error(f"Erreur dans x(t) : {err_x}")
        elif err_y:
            st.error(f"Erreur dans y(t) : {err_y}")
        else:
            fig2, ax2 = plt.subplots(figsize=(7, 6))
            ax2.plot(xv, yv, color=color_p, label=label_p, linewidth=2)
            ax2.set_title(title_p, fontsize=14)
            ax2.set_xlabel("x(t)")
            ax2.set_ylabel("y(t)")
            ax2.set_aspect("equal", adjustable="datalim")
            if grid_p:
                ax2.grid(True, alpha=0.3)
            ax2.axhline(0, color="black", linewidth=0.8)
            ax2.axvline(0, color="black", linewidth=0.8)
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)


# ─── Onglet Polaire ──────────────────────────────────────────────────────────
with tab_polar:
    col_left3, col_right3 = st.columns([1, 2])

    with col_left3:
        st.subheader("Paramètres")
        st.caption("Utiliser `theta` comme variable")
        expr_r = st.text_input("r(θ)", value="1 + cos(theta)", key="polar_r")
        theta_min, theta_max = st.slider("Plage de θ (en radians)", 0.0, 4 * np.pi, (0.0, 2 * np.pi), step=0.1, key="theta_range")
        n_pts_r = st.slider("Résolution", 200, 5000, 1000, step=100, key="n_pts_r")
        color_r = st.color_picker("Couleur", value="#2ca02c", key="color_r")
        label_r = st.text_input("Légende", value="r(θ)", key="label_r")
        title_r = st.text_input("Titre", value="Courbe polaire", key="title_r")

    with col_right3:
        theta = np.linspace(theta_min, theta_max, n_pts_r)
        ns_r = dict(SAFE_NS)
        ns_r["theta"] = theta

        rv, err_r = safe_eval(expr_r, ns_r)

        if err_r:
            st.error(f"Erreur dans r(θ) : {err_r}")
        else:
            fig3, ax3 = plt.subplots(figsize=(7, 6), subplot_kw={"projection": "polar"})
            ax3.plot(theta, rv, color=color_r, label=label_r, linewidth=2)
            ax3.set_title(title_r, fontsize=14, pad=15)
            ax3.legend(loc="upper right")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

st.divider()
st.markdown(
    "**Exemples à essayer** — "
    "Cartésien : `sin(x)/x` · `exp(-x**2)` · `x**3 - 3*x` | "
    "Paramétrique : `x=3*cos(t)`, `y=2*sin(t)` (ellipse) · `x=t-sin(t)`, `y=1-cos(t)` (cycloïde) | "
    "Polaire : `2*sin(2*theta)` (rose) · `theta` (spirale)"
)
