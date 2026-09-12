districts_df = load_district_reference_source()
if districts_df.empty or len(districts_df) == 0:
    st.error("Error loading baseline districts. Please ensure data files are present.")
    st.stop()

provinces = ["All Regions"] + sorted(list(districts_df['province'].dropna().unique()))
selected_province = st.sidebar.selectbox("Filter Province/Region", provinces)
if selected_province != "All Regions":
    filtered_districts = districts_df[districts_df['province'] == selected_province]
else:
    filtered_districts = districts_df

if filtered_districts.empty:
    filtered_districts = districts_df

district_names = sorted(filtered_districts['district'].dropna().tolist())
selected_district_name = st.sidebar.selectbox("Select Target District / Station", district_names)

matching_rows = districts_df[districts_df['district'] == selected_district_name]
if matching_rows.empty:
    selected_district_row = districts_df.iloc[0]
    selected_district_name = selected_district_row['district']
else:
    selected_district_row = matching_rows.iloc[0]

target_district_id = selected_district_row['district_id']
