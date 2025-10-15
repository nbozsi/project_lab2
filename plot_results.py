import altair as alt


def lag_chart(df, colors=["red", "green"], model_name=None, width=800, height=400):
    # Base encodings
    if "model" in df.columns:
        base = alt.Chart(df).encode(
            x=alt.X("lag:T", axis=alt.Axis(title="Lag", labelAngle=0, labelFontSize=14, titleFontSize=14, tickCount=4, format="%H:%M")),
            y=alt.Y("error:Q", axis=alt.Axis(title="Error", labelFontSize=14, titleFontSize=14)),
            color=alt.Color("Target:N", scale=alt.Scale(range=colors)).legend(
                orient="bottom", direction="vertical", labelFontSize=13, titleFontSize=13, labelLimit=600
            ),
            strokeDash=alt.StrokeDash("model:N", sort=(model_name, "Naive model")).legend(
                orient="bottom", labelFontSize=13, titleFontSize=13
            ),
        )
    else:
        base = alt.Chart(df).encode(
            x=alt.X("lag:T", axis=alt.Axis(title="Lag", labelAngle=0, labelFontSize=14, titleFontSize=14, tickCount=4, format="%H:%M")),
            y=alt.Y("error:Q", axis=alt.Axis(title="Error", labelFontSize=14, titleFontSize=14)),
            color=alt.Color("Target:N", scale=alt.Scale(range=colors)).legend(
                orient="bottom", direction="vertical", labelFontSize=13, titleFontSize=13, labelLimit=600
            ),
            strokeDash=alt.StrokeDash("error measure:N").legend(orient="bottom", labelFontSize=13, titleFontSize=13),
        )

    # Line chart
    line = base.mark_line(size=2)

    # Point markers
    points = base.mark_point(size=50, filled=False)

    # Combine
    return (points + line).properties(width=800, height=300)
