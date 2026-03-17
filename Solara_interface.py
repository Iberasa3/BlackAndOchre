import solara

# CREADO 100% CON AI, PRIMER PASO PARA GENERAR UNA INTERFAZ DE USUARIO.
count = solara.reactive(0)


@solara.component
def Page():
    # 2. El Layout (Diseño)
    with solara.Column(style={"padding": "20px", "background-color": "#f0f2f6"}):
        solara.Title("Mi Primera App en Solara")
        solara.Markdown(f"### El contador vale: **{count.value}**")

        # 3. Widgets con lógica
        solara.Button("¡Incrementar!", on_click=lambda: count.set(count.value + 1), color="primary")

        if count.value > 10:
            solara.Success("¡Has superado los 10 clicks! 🎉")

# Importante: Solara busca un componente llamado 'Page' por defecto