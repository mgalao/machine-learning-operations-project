from fraud_project.utils import *

# --------------- missing values tretament ---------------------

def impute_merch_zipcode(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # impute using most frequent zip per city
    city_zip_lookup = (
        data[~data['merch_zipcode'].isna()]
        .groupby('city')['merch_zipcode']
        .agg(lambda x: x.mode().iloc[0])
        .to_dict())
    mask_city = data['merch_zipcode'].isna() & data['city'].notna()
    data.loc[mask_city, 'merch_zipcode'] = data.loc[mask_city, 'city'].map(city_zip_lookup)

    # impute remaining using most frequent zip per state
    state_zip_lookup = (
        data[~data['merch_zipcode'].isna()]
        .groupby('state')['merch_zipcode']
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )
    mask_state = data['merch_zipcode'].isna() & data['state'].notna()
    data.loc[mask_state, 'merch_zipcode'] = data.loc[mask_state, 'state'].map(state_zip_lookup)

    # final flag for rows still missing after imputations
    data['zip_missing'] = data['merch_zipcode'].isna().astype(int)

    return data