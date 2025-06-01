from manim import *
import numpy as np


class MillerRabinVisualization(Scene):
    def construct(self):
        # ----------------- 1. Complex plane -----------------
        complex_plane = ComplexPlane(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            background_line_style={
                "stroke_color": "#1F3F73",     # dark‑blue grid
                "stroke_width": 1,
                "stroke_opacity": 0.5,
            },
        ).scale(0.8)

        title = Text("Visualizing Primality: The Miller‑Rabin Test", font_size=32)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        self.play(FadeIn(complex_plane))
        self.play(Write(Text("The Complex Plane", font_size=24).to_edge(UP)))
        self.wait(1)

        # ----------------- 2. Number under test -----------------
        test_number = Dot(complex_plane.c2p(2, 1), color="#00B3A6")
        test_label = Text("n", font_size=24).next_to(test_number, UP, buff=0.1)

        self.play(FadeOut(self.mobjects[-1]))  # remove “The Complex Plane” text
        intro = Text("Testing number n for primality", font_size=24).to_edge(UP)
        self.play(Write(intro), FadeIn(test_number), Write(test_label))
        self.wait(1)

        # ----------------- 3. Miller–Rabin transformations -----------------
        self.play(FadeOut(intro))
        transform_t = Text("Miller‑Rabin transformations", font_size=24).to_edge(UP)
        self.play(Write(transform_t))

        eq1 = Text("n^d ≡ 1 (mod n)", font_size=24).to_edge(DOWN, buff=1)
        self.play(Write(eq1))

        rotation_arc = Arc(
            arc_center=complex_plane.c2p(0, 0),
            radius=np.sqrt(5),                  # dist. |(2,1)| from origin
            start_angle=np.arctan2(1, 2),
            angle=PI,
            color="#FF7E39",
        )
        self.play(Create(rotation_arc))
        self.wait(1)

        self.play(FadeOut(eq1), FadeOut(rotation_arc))

        eq2 = Text("a^(2^r·d) ≡ -1 (mod n)", font_size=24).to_edge(DOWN, buff=1)
        self.play(Write(eq2))

        reflection_line = Line(
            complex_plane.c2p(-5, 0), complex_plane.c2p(5, 0), color="#C0C0C0"
        )
        reflected = Dot(complex_plane.c2p(2, -1), color="#00B3A6")

        self.play(Create(reflection_line))
        self.play(Transform(test_number.copy(), reflected))
        self.wait(1)

        self.play(FadeOut(eq2), FadeOut(reflection_line), FadeOut(transform_t))

        # ----------------- 4. Multiple trials -----------------
        trials_t = Text("Multiple trials increase accuracy", font_size=24).to_edge(UP)
        self.play(Write(trials_t))

        paths = VGroup()
        colors = ["#00B3A6", "#FF7E39", "#F1C40F"]  # teal, orange, yellow
        for color in colors:
            path = VMobject(color=color, stroke_width=2)
            start = complex_plane.c2p(2, 1)
            end = complex_plane.c2p(np.random.uniform(-2, 2), np.random.uniform(-2, 2))
            path.set_points_as_corners([start, end])
            paths.add(path)
            self.play(Create(path), run_time=0.5)

        self.wait(1)

        # ----------------- 5. Highlight error -----------------
        self.play(FadeOut(trials_t))
        err_t = Text("Some paths can mislead (error margin)", font_size=24).to_edge(UP)
        self.play(Write(err_t))

        incorrect = paths[1]  # orange path
        self.play(incorrect.animate.set_stroke(color=RED, width=4))
        self.wait(1)

        # ----------------- 6. Convergence with more trials -----------------
        self.play(FadeOut(err_t))
        concl_t = Text("More trials reduce error probability", font_size=24).to_edge(UP)
        self.play(Write(concl_t))

        self.play(FadeOut(incorrect))

        for _ in range(2):
            new_path = VMobject(color="#00B3A6", stroke_width=2)
            start = complex_plane.c2p(2, 1)
            mid = complex_plane.c2p(np.random.uniform(0, 3), np.random.uniform(0, 3))
            end = complex_plane.c2p(4, 0)  # convergence point
            new_path.set_points_as_corners([start, mid, end])
            self.play(Create(new_path), run_time=0.5)

        self.wait(1)

        final = Text(
            "The Miller‑Rabin test becomes more accurate with more trials",
            font_size=20,
        ).to_edge(DOWN)
        self.play(Write(final))
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


# If you want to run this file directly via `python miller_rabin_vis.py`,
# uncomment the following block and tweak the CLI flags as desired.
#

if __name__ == "__main__":
    from manim import config, cli
    config.background_color = "#1e1e1e"      # optional aesthetic tweak
    cli.main()                               # behaves like `manim` command
