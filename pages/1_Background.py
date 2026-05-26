import streamlit as st

st.set_page_config(page_title="Background")
st.header("Background", divider="red")
st.write("""
Cancer is a leading cause of death worldwide, with millions of new cases diagnosed each year. 
         Early detection and accurate risk prediction are crucial for improving patient outcomes and reducing the burden on healthcare systems now. 
         Once called "a disease of the rich", cancer has now become more prevalent than ever.
         This has largely been due to the concentration of **carcinogens** that are present in our daily lives, such as air we breathe, food we eat, and water we drink.
         As we grow older, the need for preventative care increases as well as a greater awareness to the amount of harm we present to the body.
         """)

st.header("Problem Statement", divider="violet")
st.write("How might we predict cancer risks based on an individual's living habits?")