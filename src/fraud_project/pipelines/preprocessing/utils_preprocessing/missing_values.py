from fraud_project.utils import *

# --------------- missing values tretament ---------------------

def impute_merch_zipcode(data: pd.DataFrame, mappings: Union[None, dict] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    data = data.copy()

    if mappings is None:
        # Training mode — compute mappings
        city_zip_lookup = (
            data[~data['merch_zipcode'].isna()]
            .groupby('city')['merch_zipcode']
            .agg(lambda x: x.mode().iloc[0])
            .to_dict()
        )
        state_zip_lookup = (
            data[~data['merch_zipcode'].isna()]
            .groupby('state')['merch_zipcode']
            .agg(lambda x: x.mode().iloc[0])
            .to_dict()
        )
    else:
        # Inference mode — use existing mappings
        city_zip_lookup = mappings.get("city_zip_lookup", {})
        state_zip_lookup = mappings.get("state_zip_lookup", {})

    # Impute using city mapping
    mask_city = data['merch_zipcode'].isna() & data['city'].notna()
    data.loc[mask_city, 'merch_zipcode'] = data.loc[mask_city, 'city'].map(city_zip_lookup)

    # Impute using state mapping
    mask_state = data['merch_zipcode'].isna() & data['state'].notna()
    data.loc[mask_state, 'merch_zipcode'] = data.loc[mask_state, 'state'].map(state_zip_lookup)

    # Final missing flag
    data['zip_missing'] = data['merch_zipcode'].isna().astype(int)

    if mappings is None:
        # Return mappings only in training mode
        return data, {
            "city_zip_lookup": city_zip_lookup,
            "state_zip_lookup": state_zip_lookup
        }
    else:
        return data