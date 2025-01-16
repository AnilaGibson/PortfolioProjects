Select *
From PortfolioProject..CovidDeaths$
where continent is not null
ORDER BY  3, 4

--Select *
--From PortfolioProject..CovidVaccinations$
--order by 3,4

Select location, date, total_cases, new_cases, total_deaths, population
From PortfolioProject..CovidDeaths$
ORDER BY  1, 2

-- Looking at total cases vs total deaths as a percentage  (United States) 

Select location, date, total_cases, total_deaths, (total_deaths/ total_cases)*100 as DeathPercentage
From PortfolioProject..CovidDeaths$
Where location like '%states%'
ORDER BY  1, 2

-- Looking at total cases vs population (United States) 
Select location, date, total_cases, population, (total_cases/ population)*100 as DeathPercentage
From PortfolioProject..CovidDeaths$
Where location like '%states%'
ORDER BY  1, 2

--Looking at Countries with highest infection Rate  compared to population
Select location, population, MAX(total_cases) as HighestInfectionCount, Max((total_cases/population))*100 as PercentPopluationInfected
From PortfolioProject..CovidDeaths$
Group by location, population
ORDER BY  PercentPopluationInfected desc

-- Showing Countries with the highest death count per population
Select location, MAX(cast(total_deaths as int)) as TotalDeathCount
From PortfolioProject..CovidDeaths$
Where continent is not null
Group by location 
ORDER BY  TotalDeathCount desc

-- By continent
Select continent, MAX(cast(total_deaths as int)) as TotalDeathCount
From PortfolioProject..CovidDeaths$
Where continent is not null
Group by continent 
ORDER BY  TotalDeathCount desc

--Global Numbers
Select date, SUM(new_cases)as totalCases, SUM(cast(new_deaths as int)) as totalDeath, 
SUM(cast(new_deaths as int))/SUM(new_cases)* 100 as DeathPercentage
From PortfolioProject..CovidDeaths$
Where continent is not null
Group by date
ORDER BY  1, 2

-- Looking at Total population vs vaccination
Select *
From PortfolioProject..CovidDeaths$ D
Join PortfolioProject..CovidVaccinations$ V
	On D.location = V.location
	and D.date = V.date

--Looking at total population vs vaccinations
Select D.continent, D.location, D.date, D.population, V.new_vaccinations, 
SUM(Convert(int, v.new_vaccinations)) OVER (Partition by D.location Order by D.location, D.date) as RollingPeopleVaccinated
From PortfolioProject..CovidDeaths$ D
Join PortfolioProject..CovidVaccinations$ V
	On D.location = V.location
	and D.date = V.date
where D.continent is not null
Order by 2,3



