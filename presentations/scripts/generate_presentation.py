"""
Script to generate a professional 4-slide presentation summarizing the entire
SURYOKTI Solar PV AC Power Prediction project: from initial model selection to
systematic 8-stage fine-tuning and final architecture deployment.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette
    PRIMARY = RGBColor(24, 43, 73)       # Deep Navy
    SECONDARY = RGBColor(37, 99, 235)    # Tech Blue
    ACCENT = RGBColor(217, 119, 6)       # Solar Amber / Orange
    TEXT_DARK = RGBColor(30, 41, 59)     # Slate 800
    TEXT_MUTED = RGBColor(100, 116, 139) # Slate 500
    BG_CARD = RGBColor(248, 250, 252)    # Slate 50
    BORDER_COL = RGBColor(226, 232, 240) # Slate 200
    WHITE = RGBColor(255, 255, 255)
    GREEN = RGBColor(16, 185, 129)       # Emerald

    def add_header(slide, title_text, category_text):
        # Header banner
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.7), Inches(0.6))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = PRIMARY

    def add_card(slide, left, top, width, height, bg_color=BG_CARD, border_color=BORDER_COL):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1)
        else:
            shape.line.fill.background()
        return shape

    # =========================================================================
    # SLIDE 1: Title & Project Overview
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)

    # Hero card
    add_card(s1, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9), bg_color=WHITE, border_color=BORDER_COL)

    # Accent decorative top bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.8), Inches(11.733), Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()

    # Title text
    t_box = s1.shapes.add_textbox(Inches(1.2), Inches(1.3), Inches(10.9), Inches(2.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    
    p0 = tf1.paragraphs[0]
    p0.text = "SURYOKTI SOLAR PV FORECASTING"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = SECONDARY
    p0.space_after = Pt(8)

    p1 = tf1.add_paragraph()
    p1.text = "From Model Exploration to Systematic 8-Stage Fine-Tuning"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = PRIMARY
    p1.space_after = Pt(12)

    p2 = tf1.add_paragraph()
    p2.text = "An end-to-end machine learning engineering workflow for highly accurate, robust, and physically consistent PV AC Power Output prediction."
    p2.font.size = Pt(14)
    p2.font.color.rgb = TEXT_MUTED

    # 3 Summary Pillars
    pillar_data = [
        ("1. Data & Physics Engineering", 
         "• 2,777 curated weather & generation records\n• Cyclical diurnal transformations (Hour_sin, Month_cos)\n• Physics-based solar angles & thermal stress\n• Strict zero-nighttime generation constraints",
         SECONDARY),
        ("2. Architecture Benchmarking", 
         "• Baseline models: Linear Regression, RF, XGBoost\n• Diagnosed RF overfitting (7.1% R² gap)\n• Stacking ensemble exploration (RF + XGBoost)\n• Evaluation of Linear, Ridge, & Neural Net meta-learners",
         ACCENT),
        ("3. Systematic 8-Stage Tuning", 
         "• Chronological TimeSeriesSplit validation\n• Regularized base-trees to control variance\n• Isolated 20% untouched final test holdout\n• Production deployment artifacts (.pkl, metadata)",
         GREEN)
    ]

    for i, (head, body, acc_col) in enumerate(pillar_data):
        c_left = Inches(1.2 + i * 3.75)
        c_top = Inches(3.7)
        c_w = Inches(3.45)
        c_h = Inches(2.6)

        c = add_card(s1, c_left, c_top, c_w, c_h, bg_color=BG_CARD, border_color=BORDER_COL)
        
        # Color pill on top of card
        pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.2), c_top + Inches(0.2), Inches(3.05), Inches(0.35))
        pill.fill.solid()
        pill.fill.fore_color.rgb = acc_col
        pill.line.fill.background()
        tf_pill = pill.text_frame
        p_pill = tf_pill.paragraphs[0]
        p_pill.text = head
        p_pill.font.size = Pt(11)
        p_pill.font.bold = True
        p_pill.font.color.rgb = WHITE
        p_pill.alignment = PP_ALIGN.CENTER

        tb_body = s1.shapes.add_textbox(c_left + Inches(0.2), c_top + Inches(0.65), Inches(3.05), Inches(1.8))
        tf_b = tb_body.text_frame
        tf_b.word_wrap = True
        tf_b.margin_top = tf_b.margin_left = tf_b.margin_right = tf_b.margin_bottom = 0
        p_b = tf_b.paragraphs[0]
        p_b.text = body
        p_b.font.size = Pt(11)
        p_b.font.color.rgb = TEXT_DARK
        p_b.line_spacing = 1.25

    # =========================================================================
    # SLIDE 2: Initial Model Exploration & Overfitting Diagnosis
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Phase 1: Baseline Exploration & Architecture Benchmarking", "Exploratory Analysis")

    # Left Column: Insights Card
    left_card = add_card(s2, Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.5))
    tb_l = s2.shapes.add_textbox(Inches(1.05), Inches(1.7), Inches(5.3), Inches(5.1))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    
    p = tf_l.paragraphs[0]
    p.text = "Key Observations Across Model Families"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(10)

    points = [
        ("Linear Regression (Baseline):", "High error (MAE 158.56 kW, R² 0.7297). Unable to map nonlinear solar atmospheric dynamics and solar zenith geometry."),
        ("Random Forest (Deep Trees):", "High raw accuracy (R² 0.9159), but suffered severe memorization (Train R² 0.9871, gap 7.11%). Risky for real-world unseen seasonal shifts."),
        ("XGBoost Regressor:", "Balanced learning (R² 0.9200, MAE 55.30 kW) with lower generalization gap (2.45%) and strong gradient-guided tree splits."),
        ("Stacking Ensemble Formulation:", "Stacked Random Forest + XGBoost with LinearRegression meta-learner. Achieved R² ~0.9200 and MAE 54.14 kW, proving multi-model synergy.")
    ]

    for title, desc in points:
        pt = tf_l.add_paragraph()
        pt.text = "• " + title + " "
        pt.font.bold = True
        pt.font.size = Pt(11)
        pt.font.color.rgb = SECONDARY
        
        # description
        run = pt.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_DARK
        pt.space_after = Pt(8)

    # Right Column: Visual chart & Metric Table
    add_card(s2, Inches(6.8), Inches(1.5), Inches(5.733), Inches(5.5))
    
    # Embed model comparison plot if available
    img_candidates = [
        os.path.join("plots", "energy_prediction", "hybrid_model_comparison.png"),
        os.path.join("Plots", "energy_prediction", "hybrid_model_comparison.png"),
        os.path.join("plots", "energy_prediction", "model_comparison.png"),
        os.path.join("Plots", "energy_prediction", "model_comparison.png"),
        os.path.join("Plots", "hybrid_model_comparison.png"),
        os.path.join("plots", "hybrid_model_comparison.png"),
        os.path.join("Plots", "model_comparison.png"),
        os.path.join("plots", "model_comparison.png"),
    ]
    img_path = next((p for p in img_candidates if os.path.exists(p)), None)
    
    if img_path and os.path.exists(img_path):
        s2.shapes.add_picture(img_path, Inches(7.0), Inches(1.65), Inches(5.333), Inches(2.6))

    # Metric benchmark table below image
    rows = [
        ["Model Architecture", "MAE (kW)", "RMSE (kW)", "Holdout R²", "Overfit Gap"],
        ["Linear Regression", "158.56", "189.61", "0.7297", "0.0149"],
        ["Random Forest (alone)", "53.98", "105.74", "0.9159", "0.0711 (High)"],
        ["XGBoost (alone)", "55.30", "103.18", "0.9200", "0.0245"],
        ["Hybrid (RF + XGB -> LR)", "54.14", "103.19", "0.9200", "0.0487"],
        ["Unified 3-Stack (RF+XGB->NN)", "52.26", "101.18", "0.9230", "0.0192"]
    ]

    tbl_shape = s2.shapes.add_table(len(rows), len(rows[0]), Inches(7.0), Inches(4.4), Inches(5.333), Inches(2.3))
    table = tbl_shape.table
    table.columns[0].width = Inches(2.133)
    table.columns[1].width = Inches(0.8)
    table.columns[2].width = Inches(0.8)
    table.columns[3].width = Inches(0.8)
    table.columns[4].width = Inches(0.8)

    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p_c = cell.text_frame.paragraphs[0]
            p_c.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            p_c.font.size = Pt(8.5)
            if r_idx == 0:
                p_c.font.bold = True
                p_c.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = PRIMARY
            else:
                p_c.font.color.rgb = TEXT_DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 0 else BG_CARD

    # =========================================================================
    # SLIDE 3: The 8-Stage Fine-Tuning Pipeline
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Phase 2: Systematic 8-Stage Fine-Tuning Pipeline", "Model Optimization")

    # Flowchart / Process steps laid out in 2 rows of 4 cards
    stages = [
        ("Stage 1: Baseline 3-Stack", "RF + XGBoost -> MLP\nR² = 0.9052, MAE = 68.40 kW\nEstablished baseline on chronological dev split.", SECONDARY),
        ("Stage 2: 2-Stack vs 3-Stack", "Controlled meta comparison:\nRidgeCV (R² 0.9128) beat MLP (0.9052) under temporal CV.", PRIMARY),
        ("Stage 3: Tune XGBoost", "Evaluated depth 3, 4, 6.\nSelected XGB_Compact (depth=4, lr=0.05, n=250): CV R² 0.9036.", ACCENT),
        ("Stage 4: Tune Random Forest", "Constrained tree depth to prevent memorization.\nSelected RF_Regularized (depth=7, leaf=5, n=250): CV R² 0.9127.", GREEN),
        ("Stage 5: Tune Neural Net", "Tested 3 architectures:\nSelected NN_Balanced (64->32, alpha=0.05) achieving CV R² 0.9070.", SECONDARY),
        ("Stage 6: Compare Meta-Learners", "Benchmarked RidgeCV, LinearReg, ElasticNet, GradBoost, & MLP.\nRidgeCV ranked #1 (CV R² 0.9126).", PRIMARY),
        ("Stage 7: Temporal Generalization", "5-Fold TimeSeriesSplit validation.\nConfirmed RidgeCV had lowest variance across seasons (±0.0086).", ACCENT),
        ("Stage 8: Final Fit & Test", "Selected RF + XGB -> RidgeCV.\nEvaluated once on 20% untouched holdout test: R² = 91.15%.", GREEN)
    ]

    card_w = Inches(2.78)
    card_h = Inches(2.5)

    for idx, (title, desc, accent) in enumerate(stages):
        row = idx // 4
        col = idx % 4
        x = Inches(0.8 + col * 2.98)
        y = Inches(1.5 + row * 2.8)

        card = add_card(s3, x, y, card_w, card_h, bg_color=WHITE, border_color=BORDER_COL)
        
        # Step tag
        step_bar = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, card_w, Inches(0.08))
        step_bar.fill.solid()
        step_bar.fill.fore_color.rgb = accent
        step_bar.line.fill.background()

        tb = s3.shapes.add_textbox(x + Inches(0.15), y + Inches(0.18), card_w - Inches(0.3), card_h - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_top = tf.margin_left = tf.margin_right = tf.margin_bottom = 0

        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.bold = True
        p_t.font.size = Pt(11)
        p_t.font.color.rgb = PRIMARY
        p_t.space_after = Pt(6)

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(9.5)
        p_d.font.color.rgb = TEXT_DARK
        p_d.line_spacing = 1.2

    # =========================================================================
    # SLIDE 4: Final Architecture & Deployment Verification
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Phase 3: Final Architecture & Production Verification", "Results & Deployment")

    # Left Container: Final Architecture Block
    add_card(s4, Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.5), bg_color=WHITE)
    tb_arch = s4.shapes.add_textbox(Inches(1.05), Inches(1.7), Inches(5.3), Inches(5.1))
    tf_arch = tb_arch.text_frame
    tf_arch.word_wrap = True

    p = tf_arch.paragraphs[0]
    p.text = "Selected Production Architecture"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(10)

    arch_sections = [
        ("Level 1 Base Learners:", 
         "• RF_Regularized (n=250, max_depth=7, min_samples_leaf=5, max_features=0.75)\n• XGB_Compact (n=250, max_depth=4, learning_rate=0.05, reg_lambda=2.0)"),
        ("Feature Passthrough Integration:", 
         "• Solar Physics & Cyclical Features: Hour_sin/cos, Month_sin/cos, Wind_Dir_sin/cos\n• Effective_GHI (SZ_Angle cosine projection) & Thermal_Stress index"),
        ("Level 2 Meta-Learner:", 
         "• RidgeCV with L2 regularization (alphas=[0.01, 0.1, 1.0, 10.0])\n• Optimal temporal stability across all 5 chronological validation splits"),
        ("Physics-Based Output Constraint:", 
         "• SolarOutputRegressor wrapper enforcing zero production whenever GHI <= 0\n• Strict non-negative clipping eliminating night-time prediction drift")
    ]

    for title, desc in arch_sections:
        pt = tf_arch.add_paragraph()
        pt.text = title
        pt.font.bold = True
        pt.font.size = Pt(11)
        pt.font.color.rgb = SECONDARY
        
        pd = tf_arch.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(9.5)
        pd.font.color.rgb = TEXT_DARK
        pd.space_after = Pt(8)

    # Right Container: Verification Metrics & Exported Artifacts
    add_card(s4, Inches(6.8), Inches(1.5), Inches(5.733), Inches(5.5), bg_color=WHITE)

    # Metric Highlights Card
    m_box = add_card(s4, Inches(7.0), Inches(1.7), Inches(5.333), Inches(2.2), bg_color=BG_CARD)
    tb_m = s4.shapes.add_textbox(Inches(7.2), Inches(1.8), Inches(4.9), Inches(2.0))
    tf_m = tb_m.text_frame
    tf_m.word_wrap = True

    pm_title = tf_m.paragraphs[0]
    pm_title.text = "Untouched Final Chronological Test Set (556 Rows)"
    pm_title.font.bold = True
    pm_title.font.size = Pt(13)
    pm_title.font.color.rgb = PRIMARY
    pm_title.space_after = Pt(8)

    # Metric badges
    badges = [
        ("R² Accuracy", "91.15%", GREEN),
        ("Mean Abs. Error", "64.76 kW", SECONDARY),
        ("Root Mean Sq. Error", "113.50 kW", ACCENT)
    ]
    for b_title, b_val, b_col in badges:
        p_b = tf_m.add_paragraph()
        p_b.text = f"{b_title}: "
        p_b.font.size = Pt(11)
        p_b.font.color.rgb = TEXT_DARK
        r = p_b.add_run()
        r.text = b_val
        r.font.bold = True
        r.font.color.rgb = b_col

    # Exported Artifacts & Deployment Readiness
    tb_art = s4.shapes.add_textbox(Inches(7.0), Inches(4.1), Inches(5.333), Inches(2.7))
    tf_art = tb_art.text_frame
    tf_art.word_wrap = True

    pa_title = tf_art.paragraphs[0]
    pa_title.text = "Production Artifacts & Deliverables"
    pa_title.font.bold = True
    pa_title.font.size = Pt(13)
    pa_title.font.color.rgb = PRIMARY
    pa_title.space_after = Pt(8)

    art_points = [
        ("Model Artifact:", "models/fine_tuned_final_stack.pkl (Fitted on development partition ready for deployment)"),
        ("Features Schema:", "models/fine_tuned_features.pkl (Strict 18-feature schema aligned with physics engineering)"),
        ("Audit Metadata:", "models/fine_tuned_metadata.json (Full reproducibility audit trail with exact metrics & parameters)"),
        ("Inference Script:", "Prediction.py (Standalone prediction script tested on 20% holdout & single-sample demo)")
    ]

    for title, desc in art_points:
        pt = tf_art.add_paragraph()
        pt.text = "✔ " + title + " "
        pt.font.bold = True
        pt.font.size = Pt(10)
        pt.font.color.rgb = GREEN
        
        run = pt.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_DARK
        pt.space_after = Pt(6)

    # Save presentation
    out_dir = "Presentation"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "SURYOKTI_Model_Selection_and_FineTuning.pptx")
    prs.save(out_path)
    final_dir = os.path.join("presentations", "final")
    if os.path.exists(final_dir):
        final_path = os.path.join(final_dir, "SURYOKTI_Model_Selection_and_FineTuning.pptx")
        prs.save(final_path)
    print(f"Successfully generated PowerPoint presentation at: {out_path}")
    return out_path

if __name__ == "__main__":
    create_deck()

