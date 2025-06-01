from manim import *
import numpy as np


class MillerRabinVisualization(Scene):
    def construct(self):
        # --------------------------------------------------
        # 1. Complex plane
        # --------------------------------------------------
        plane = ComplexPlane(
            x_range=[-5, 5, 1],
            y_range=[0, 4.5, 0.5],
            background_line_style={
                "stroke_color": "#1F3F73",
                "stroke_width": 1,
                "stroke_opacity": 0.5,
            },
        ).scale(0.8)

        self.play(Create(plane, run_time=2))

        # --------------------------------------------------
        # 2. Title & subtitle
        # --------------------------------------------------
        title = Text("Visualizing Primality: The Miller‑Rabin Test", font_size=32)
        subtitle = Text("- A Closer Look at Error Margins -", font_size=24)
        titles = VGroup(title, subtitle).arrange(DOWN, buff=0.3).to_edge(UP)

        self.play(Write(titles))
        self.wait(2)
        self.play(FadeOut(titles))

        # --------------------------------------------------
        # 3. Intro sentence
        # --------------------------------------------------
        intro = Text(
            "The Miller‑Rabin test offers an efficient\n"
            "probabilistic check for primality by applying\n"
            "modular‑exponentiation transforms.",
            font_size=20,
            line_spacing=1.2,
        ).to_edge(UP)
        self.play(Write(intro))
        self.wait(2)
        self.play(FadeOut(intro))

        # --------------------------------------------------
        # 4. Element 1 – start with a number n
        # --------------------------------------------------
        start_t = Text("Start with a number  n", font_size=22).to_edge(UP)
        n_dot = Dot(plane.c2p(2, 1), color="#00B3A6")

        self.play(Write(start_t), FadeIn(n_dot))
        self.wait(1)

        # --------------------------------------------------
        # 5. Element 2 – repeated squaring
        # --------------------------------------------------
        self.play(FadeOut(start_t))
        step1_t = Text(
            "Repeated squaring drives  n  toward  n^d ≡ 1 (mod n)",
            font_size=22,
        ).to_edge(UP)

        arc = ArcBetweenPoints(
            plane.c2p(2, 1),
            plane.c2p(-1, 3),
            angle=PI / 3,
            stroke_color=BLUE,
            stroke_width=2,
        )

        self.play(Write(step1_t), Create(arc))
        self.wait(1)
        self.play(FadeOut(step1_t), FadeOut(arc))

        # --------------------------------------------------
        # 6. Element 3 – potential error path
        # --------------------------------------------------
        err_t = Text(
            "Some witnesses escape early  →  potential error",
            font_size=24,
        ).to_edge(UP)

        err_path = VMobject(stroke_color=RED, stroke_width=3)
        err_path.set_points_as_corners([plane.c2p(2, 1), plane.c2p(4, 4)])

        self.play(Write(err_t), Create(err_path))
        self.wait(1)
        self.play(FadeOut(err_t), FadeOut(err_path))

        # --------------------------------------------------
        # 7. Element 4 – more trials
        # --------------------------------------------------
        more_t = Text("More trials reduce error probability", font_size=28).to_edge(UP)
        self.play(Write(more_t))

        for _ in range(3):
            path = VMobject(stroke_color=GREEN, stroke_width=2)
            mid = plane.c2p(np.random.uniform(-2, 2), np.random.uniform(1, 4))
            path.set_points_as_corners([plane.c2p(2, 1), mid, plane.c2p(0, 4)])
            self.play(Create(path), run_time=0.4)

        self.wait(1)

        # --------------------------------------------------
        # 8. Conclusion (LaTeX box)
        # --------------------------------------------------
        concl = MathTex(
            r"\boxed{\text{Miller–Rabin} \;\Longrightarrow\; "
            r"\text{Efficient Primality Test}\;(\text{Probabilistic Error})}",
            font_size=32,
        ).to_edge(DOWN)

        self.play(Write(concl))
        self.wait(3)

        # --------------------------------------------------
        # 9. Fade everything out
        # --------------------------------------------------
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


# ------------------------------------------------------------------
# Command‑line usage
#   Preview (quick, low‑res):   manim -pql miller_rabin_vis.py MillerRabinVisualization
#   High‑quality render:        manim -pqh miller_rabin_vis.py MillerRabinVisualization
# ------------------------------------------------------------------

if __name__ == "__main__":
    from manim import config, cli
    config.background_color = "#1e1e1e"      # optional aesthetic tweak
    cli.main() 