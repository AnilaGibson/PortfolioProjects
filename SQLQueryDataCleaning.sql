-- Cleaning data

Select * 
from PortfolioProject2.dbo.Housing

-- Standardize Date Format

Select SaleDate, Convert(Date, SaleDate)
from PortfolioProject2.dbo.Housing

Update Housing
Set SaleDate = CONVERT(Date,SaleDate)

ALTER TABLE Housing
Add SaleDateNew Date;

update Housing
SET SaleDateNew = Convert(Date,SaleDate)


-- Populate Property Address data

Select *
from PortfolioProject2.dbo.Housing
Where PropertyAddress is null 

Select A.ParcelID, A.PropertyAddress, B.ParcelID, B.PropertyAddress, ISNULL(A.PropertyAddress, B.PropertyAddress)
from PortfolioProject2.dbo.Housing  A
JOIN PortfolioProject2.dbo.Housing B
	on A.ParcelID = B.ParcelID
	AND A.[UniqueID ] <> b.[UniqueID ]
Where A.PropertyAddress is null

Update A
Set PropertyAddress = ISNULL(A.PropertyAddress, B.PropertyAddress)
from PortfolioProject2.dbo.Housing  A
JOIN PortfolioProject2.dbo.Housing B
	on A.ParcelID = B.ParcelID
	AND A.[UniqueID ] <> b.[UniqueID ]
Where A.PropertyAddress is null


-- Breaking out Address into Individual Columns( Adrdress, City, State)

Select PropertyAddress
from PortfolioProject2.dbo.Housing

Select
SUBSTRING(PropertyAddress, 1, CHARINDEX(', ', PropertyAddress) -1) as Address
, SUBSTRING(PropertyAddress, CHARINDEX(', ', PropertyAddress) +1, len(PropertyAddress)) as Address
from PortfolioProject2.dbo.Housing

ALTER TABLE Housing
Add PropertySplitAddress Nvarchar(255);

update Housing
SET PropertySplitAddress = SUBSTRING(PropertyAddress, 1, CHARINDEX(', ', PropertyAddress) -1)

ALTER TABLE Housing
Add PropertySplitCity Nvarchar(255);

update Housing
SET PropertySplitCity = SUBSTRING(PropertyAddress, CHARINDEX(', ', PropertyAddress) +1, len(PropertyAddress))




Select OwnerAddress
From Housing

Select
PARSENAME(REPLACE(OwnerAddress, ',', '.'),  3)
,PARSENAME(REPLACE(OwnerAddress, ',', '.'),  2)
,PARSENAME(REPLACE(OwnerAddress, ',', '.'),  1)
From Housing

ALTER TABLE Housing
Add OwnerSplitAddress Nvarchar(255);

update Housing
SET OwnerSplitAddress = PARSENAME(REPLACE(OwnerAddress, ',', '.'),  3)


ALTER TABLE Housing
Add OwnerSplitCity Nvarchar(255);

update Housing
SET OwnerSplitCity = PARSENAME(REPLACE(OwnerAddress, ',', '.'),  2)


ALTER TABLE Housing
Add OwnerSplitState Nvarchar(255);

update Housing
SET OwnerSplitState = PARSENAME(REPLACE(OwnerAddress, ',', '.'),  1)


-- Change Y and N to Yes and No in 'Solid as Vacant' field

Select Distinct(SoldAsVacant), Count(SoldAsVacant)
from Housing
group by SoldAsVacant
ORDER BY 2

Select SoldAsVacant, 
Case When SoldAsVacant = 'Y' THEN 'Yes'
	When SoldAsVacant = 'N' THEN 'No'
	ELSE SoldAsVacant
	END
from Housing

update Housing
SET SoldAsVacant = Case When SoldAsVacant = 'Y' THEN 'Yes'
	When SoldAsVacant = 'N' THEN 'No'
	ELSE SoldAsVacant
	END

-- Remove Dupicates
WITH ROWCTE AS(
Select *, 
 ROW_NUMBER() OVER(
 PARTITION BY ParcelID, 
				PropertyAddress, 
				SalePrice, 
				SaleDate, 
				LegalReference
				ORDER BY 
					UniqueID
					) row_num
From Housing
)
Select *
from ROWCTE
WHERE row_num > 1
Order by PropertyAddress

-- Delete Unused Columns
Alter Table Housing
Drop Column OwnerAddress, TaxDistrict, PropertyAddress

Alter Table Housing
Drop Column SaleDate