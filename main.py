import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score


def weighted_median(values, weights):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, 0.5 * c[-1])]]

def regionPlot(region, df_gdp, df_temp, year = 2010):
    '''
    Creates country specific regional GDP-per-capita vs regional Temperature correlation plots for a year
    and calculates their slopes from the DOSE dataset (Kotz et al.)
    :param region: country to produce graph for
    :param df_gdp: DOSE dataframe with GDP data
    :param df_temp: DOSE dataframe with temperature data
    :param year: year to take regional GDP and Temperature data from
    :return: prints graphs 
    '''

    gdp = []
    pop = []
    temp = []

    for subregion in df_gdp[df_gdp['GID_0'] == region]['GID_1'].unique():

        gdp_val = np.nanmean(df_gdp[(df_gdp['GID_1'] == subregion) & (df_gdp['year'].isin([year for year in range(year-5,year+5)]))]['grp_pc_usd_2015'].values)
        pop_val = np.nanmean(df_gdp[(df_gdp['GID_1'] == subregion) & (
            df_gdp['year'].isin([year for year in range(year - 5, year + 5)]))]['pop'].values)
        temp_val = np.nanmean(df_temp[(df_temp['GID_1'] == subregion) & (df_temp['year'].isin([year for year in range(year-5,year+5)]))]['Tmean'].values)

        if gdp_val == gdp_val and temp_val == temp_val and pop_val == pop_val:

            gdp.append(gdp_val)
            pop.append(pop_val)
            temp.append(temp_val)

        else:

            print(f'Excluded {subregion}')

        try:
            country_name = df_gdp[(df_gdp['GID_1'] == subregion) & (df_gdp['year'].isin([year for year in range(year - 5, year + 5)]))]['country'].values[0]
        except:
            country_name = region

    population_sizes = [popu / 50000 for popu in pop]  # Scaling down population for dot sizes

    temperatures_arr = np.array(temp).reshape(-1, 1)
    gdp_values_arr = np.array(gdp)
    population_arr = np.array(population_sizes)

    # Weight GDP values by population size
    weights = population_arr / np.sum(population_arr)

    model = LinearRegression()
    model.fit(temperatures_arr, gdp_values_arr,  sample_weight=weights)

    # Predict GDP values based on the fitted model
    predicted_gdp_values = model.predict(temperatures_arr)

    # Scatter plot
    plt.figure(figsize=(10, 8))

    # Plot each point with temperature on x-axis, GDP on y-axis, and population size as dot size
    plt.scatter(temp, gdp, s=population_sizes, alpha=0.5)
    plt.plot(temp, predicted_gdp_values, color='red', label='Linear Regression')

    # Add labels and title
    plt.xlabel('Annual Mean Temperature (°C)')
    plt.ylabel('GDP per capita (2015 dollars)')
    plt.title(f'Temperature vs. GDP in {country_name}')

    r_squared = r2_score( gdp_values_arr, temperatures_arr)
    plt.text(18, 2100, f'R-squared = {r_squared:.2f}', fontsize=12)

    # Display plot
    plt.grid(True)  # Add grid for better readability (optional)
    plt.show()


def worldPlot(df_gdp, df_temp, year = 2010, withSlope= True):
    '''
    Creates worldwide regional GDP-per-capita vs regional Temperature correlation plots for a year
    and calculates their slopes from the DOSE dataset (Kotz et al.)
    :param df_gdp: DOSE dataframe with GDP data
    :param df_temp: DOSE dataframe with temperature data
    :param year: year to take regional GDP and Temperature data from
    :param withSlope: when true also creates a graph with each regions slopes
    :return: prints graphs
    '''

    gdp = []
    pop = []
    temp = []
    name = []

    if withSlope:
        slope = []
        intercept = []
        populations = []

    for region in df_gdp['GID_0'].unique():

        try:
            temp_reg = []
            gdp_reg = []
            pop_reg = []
            for subregion in df_gdp[df_gdp['GID_0'] == region]['GID_1'].unique():

                gdp_val = np.nanmean(df_gdp[(df_gdp['GID_1'] == subregion) & (df_gdp['year'].isin([year for year in range(year-5,year+5)]))]['grp_pc_usd_2015'].values)

                pop_val = np.nanmean(df_gdp[(df_gdp['GID_1'] == subregion) & (
                    df_gdp['year'].isin([year for year in range(year - 5, year + 5)]))]['pop'].values)
                temp_val = np.nanmean(df_temp[(df_temp['GID_1'] == subregion) & (df_temp['year'].isin([year for year in range(year-5,year+5)]))]['Tmean'].values)
                name_val = df_gdp[(df_gdp['GID_1'] == subregion) & (
                    df_gdp['year'].isin([year for year in range(year - 5, year + 5)]))]['region'].values[0]


                if gdp_val == gdp_val and temp_val == temp_val and pop_val == pop_val:
                    gdp.append(gdp_val)
                    pop.append(pop_val)
                    temp.append(temp_val)
                    gdp_reg.append(gdp_val)
                    pop_reg.append(pop_val)
                    temp_reg.append(temp_val)
                    name.append(name_val)
                else:
                    print(f'Excluded {subregion}')
        except:

            print(f'Excluded {region}')

        if withSlope:

            if pop_reg != [] and temp_reg != [] and gdp_reg != []:

                model = LinearRegression()
                weights = pop_reg / np.sum(pop_reg)
                model.fit(np.array(temp_reg).reshape(-1, 1), np.array(gdp_reg), sample_weight=weights)
                slope.append(model.coef_[0])
                populations.append(np.sum(np.array(pop_reg)))
                intercept.append(model.intercept_)





    population_sizes = [popu / 500000 for popu in pop]  # Scaling down population for dot sizes

    temperatures_arr = np.array(temp).reshape(-1, 1)
    order = np.argsort(np.array(temp))
    gdp_values_arr = np.array(gdp)
    gdp_values_arr_log = np.log(np.array(gdp))
    population_arr = np.array(population_sizes)

    # Weight GDP values by population size
    weights = population_arr / np.sum(population_arr)

    model = LinearRegression()
    model.fit(temperatures_arr, gdp_values_arr,  sample_weight=weights)

    # Predict GDP values based on the fitted model
    predicted_gdp_values = model.predict(temperatures_arr)

    model_log = LinearRegression()
    model_log.fit(temperatures_arr, gdp_values_arr_log, sample_weight=weights)

    # Predict GDP values based on the fitted model
    predicted_gdp_values_log = model_log.predict(temperatures_arr)

    predicted_gdp_values_log = np.exp(predicted_gdp_values_log)

    # Scatter plot
    plt.figure(figsize=(10, 8))  # Adjust figure size as needed

    # Plot each point with temperature on x-axis, GDP on y-axis, and population size as dot size
    plt.scatter(temp, gdp, s=population_sizes, alpha=0.5)
    #plt.yscale('log')
    plt.plot(np.array(temp)[order], predicted_gdp_values[order], color='red', label='Linear Regression')
    plt.plot(np.array(temp)[order], predicted_gdp_values_log[order], color='blue', label='Exponential Regression')

    # Add labels and title
    plt.xlabel('Annual Mean Temperature (°C)')
    plt.ylabel('GDP per capita (2015 dollars)')
    plt.title(f'Temperature vs. GDP in all world regions')

    r_squared = r2_score(gdp_values_arr, temperatures_arr)
    plt.text(18, 2100, f'R-squared = {r_squared:.2f}', fontsize=12)

    # Display plot
    plt.grid(True)  # Add grid for better readability
    plt.show()
    print(f'Slope Model {model.coef_[0]}')

    if withSlope:
        plt.figure(figsize=(10, 8))
        plt.scatter([i for i in range(len(slope))], slope)
        plt.plot([i for i in range(len(slope))], [0 for i in range(len(slope))], color='blue')
        plt.xlabel('Slope')
        plt.ylabel('Intercept')
        plt.title(f'Slope and intercept of interregional Regression')
        plt.show()

        print(f'Average slope: {np.mean(np.array(slope))}')
        print(f'Average slope weighted by pop size: {np.average(np.array(slope), weights = np.array(populations))}')
        print(f'Median slope: {np.median(np.array(slope))}')
        print(f'Median slope weighted by pop size {weighted_median(np.array(slope), np.array(populations))}')
        print(f'Full population considered: {np.sum(np.array(populations))}')



if __name__ == '__main__':
    df_gdp = pd.read_csv('/Users/niklasschwind/Downloads/DOSE_V2.csv')
    df_temp = pd.read_csv('/Users/niklasschwind/Downloads/DOSEV2_W5E5D_full.csv')
    #regionPlot('TUR', df_gdp, df_temp) #'KEN',LBY, NER, NGA, SAU, AUS, PER, PHL, POL, SUD, SWE, TUR, ZAF
    worldPlot(df_gdp, df_temp)

