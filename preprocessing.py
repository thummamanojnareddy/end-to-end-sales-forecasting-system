import pandas as pd


def load_data(path):

    # Read CSV file
    df = pd.read_csv(path)

    # Rename columns
    df.columns = ['state', 'date', 'sales', 'category']

    # Handle mixed date formats
    df['date'] = pd.to_datetime(
        df['date'],
        format='mixed',
        dayfirst=True
    )

    # Clean sales column
    df['sales'] = (
        df['sales']
        .astype(str)
        .str.replace(',', '')
        .str.strip()
    )

    # Convert sales column to numeric
    df['sales'] = pd.to_numeric(
        df['sales'],
        errors='coerce'
    )

    # Sort data
    df = df.sort_values(['state', 'date'])

    return df


def handle_missing_dates(df):

    final_df = []

    for state in df['state'].unique():

        state_df = df[df['state'] == state].copy()

        # Create continuous weekly dates
        full_dates = pd.date_range(
            start=state_df['date'].min(),
            end=state_df['date'].max(),
            freq='W'
        )

        # Reindex data
        state_df = (
            state_df
            .set_index('date')
            .reindex(full_dates)
        )

        # Restore state column
        state_df['state'] = state

        # Fill missing sales values
        state_df['sales'] = (
            state_df['sales']
            .interpolate()
        )

        # Reset index
        state_df = state_df.reset_index()

        # Rename date column
        state_df.rename(
            columns={'index': 'date'},
            inplace=True
        )

        final_df.append(state_df)

    return pd.concat(final_df)