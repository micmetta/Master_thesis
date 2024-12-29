###############################################################################################################################################################################################################
# - Script containing all the functions that allow you to create tables, queries and answers that I used to create the dataset 'BioInfoDataset_v3.json'.
# - All these functions are called by the script 'creazioneDatasetBioInfoDB_v3.py'.
###############################################################################################################################################################################################################

###############################################################################################################################################################################################################
###############################################################################################################################################################################################################
# REMEMBER:

# 1) The following fields are present in the virus table but are not found with the same name in detailedstrain, so I did this name mapping:

# vorganismtype (virus) -> organismtype (detailedstrain)
# vriskgroup (virus) -> riskgroup (detailedstrain)
# symptomatologyisolationhost (virus) -> symptomatologyonisolationhost (detailedstrain)
###############################################################################################################################################################################################################
###############################################################################################################################################################################################################


################################################################################################################################################################
# - TABLE
def get_detailedstrain_table():
    only_table_detailedstrain =  "CREATE TABLE detailedstrain ( strainid number PRIMARY KEY, collection number, accessionnumber text, gmo boolean, gmoconstructioninfo text, registeredcollection boolean, collectiondate text, isolationdate text, identificationtechnique text, depositdate text, otherdenominations text, typestrain boolean, inclusiondate text, remarks text, pathogenicity text, beforenagoya boolean, availablefordis boolean, otherccnumbers text, commentontaxonomy text, mutantinformation text, userestrictions number, nagoyaconditions number, geoorigin number, cacronym text, confirmed boolean, organismtype text, riskgroup number, euquarantine boolean, strain number, recommendedtemperature number, dualuse boolean, substrate text, testedtgmin number, testedtgmax number, morganismtype text, sexualstate text, mtafile text, prodofmetabolites text, genus text, species text, meuquarantine boolean, isolationhabitat text, status text, infrasubspecificnames text, qps boolean, axenic boolean, genotype text, mriskgroup number, historyofdeposit text, enzymeproduction text, applications text, plasmids text, plasmidscollectionsfields text, ploidy text, interspecifichybrid boolean, scientificname text, ineva boolean, inmirri boolean, othernames text, isolationhost text, name text, abbreviation text, propagationhost text, cultivar text, lyticcicle text, pathotypeserotypetype text, storageconditions text, symptomatologyonisolationhost text, transmissionby text, contamination boolean, infectivitytested boolean, restrictions text, nagoyaprotocol text );"
    return only_table_detailedstrain
################################################################################################################################################################


################################################################################################################################################################
# Questions and Queries of category C1: (only on detailedstrain table)

# Questions:
def get_C1_Q1():
    return "Count the number of strains with identificationtechnique equal to ELISA."
def get_C1_Q2():
    return "Count the number of strains with organismtype equal to 'Virus'."
def get_C1_Q3():
    return "Count the number of strains with pathogenicity equal to 'Pathogen for Plants'."
def get_C1_Q4():
    return "Count the strains where the othernames field is equal to 'Iris yellow spot virus'."
def get_C1_Q5():
    return "Select all strains with gmoconstructioninfo equal to 'Gene Knockout'."
def get_C1_Q6():
    return "Count the number of strains with scientificname equal to 'Bean common mosaic virus'."
def get_C1_Q7():
    return "Select all strains where the typestrain is equal to true."
def get_C1_Q8():
    return "Count the strains where the cultivar field is null."
def get_C1_Q9():
    return "Select all strains with accessionnumber equal to 'ABC123'."
def get_C1_Q10():
    return  "Select all strains with propagationhost equal to 'Solanaceae/Nicotiana/benthamiana'."


# Answers (Queries):
def get_C1_A1():
    return "SELECT COUNT(*) FROM detailedstrain WHERE identificationtechnique = 'ELISA';"
def get_C1_A2():
    return "SELECT COUNT(*) FROM detailedstrain WHERE organismtype = 'Virus';"
def get_C1_A3():
    return "SELECT COUNT(*) FROM detailedstrain WHERE pathogenicity = 'Pathogen for Plants';"
def get_C1_A4():
    return "SELECT COUNT(*) FROM detailedstrain WHERE othernames = 'Iris yellow spot virus';"
def get_C1_A5():
    return "SELECT * FROM detailedstrain WHERE gmoconstructioninfo = 'Gene Knockout';"
def get_C1_A6():
    return "SELECT COUNT(*) FROM detailedstrain WHERE scientificname = 'Bean common mosaic virus';"
def get_C1_A7():
    return "SELECT * FROM detailedstrain WHERE typestrain = true;"
def get_C1_A8():
    return "SELECT COUNT(*) FROM detailedstrain WHERE cultivar is null;"
def get_C1_A9():
    return "SELECT * FROM detailedstrain WHERE accessionnumber = 'ABC123';"
def get_C1_A10():
    return "SELECT * FROM detailedstrain WHERE propagationhost = 'Solanaceae/Nicotiana/benthamiana';"
################################################################################


################################################################################
# Questions and Queries of category C2: (only on detailedstrain table)

# Questions:
def get_C2_Q1():
    return "Count the number of strains with identificationtechnique equal to 'ELISA' and gmoconstructioninfo equal to 'Gene Knockout'."
def get_C2_Q2():
    return "Select all strains where pathogenicity is 'Pathogen for Plants' and typestrain is true."
def get_C2_Q3():
    return "Count the number of strains where remarks is 'FPO 8' or accessionnumber is 'ABC123'."
def get_C2_Q4():
    return "Select all strains where otherdenominations is 'MVPC1 10' and identificationtechnique is 'PCR'."
def get_C2_Q5():
    return "Count the number of strains where otherccnumbers is 'CC123' or commentontaxonomy is 'Lactobacillus plantarum'."
def get_C2_Q6():
    return "Select all strains where pathogenicity is 'Pathogen for Plants' or gmoconstructioninfo is 'Gene Knockout'."
def get_C2_Q7():
    return "Count the number of strains where name is 'Tobacco ringspot virus' and riskgroup is 3."
def get_C2_Q8():
    return  "Select all strains where name is Bean common mosaic virus or organismtype is Virus."
def get_C2_Q9():
    return "Count the number of strains where pathotypeserotypetype is 'diverso' and scientificname is 'DENV-1'."
def get_C2_Q10():
    return "Select all strains where name is 'Zika' and isolationhost is 'Gesneriaceae/Sinningia/speciosa'."

# Answers (Queries):
def get_C2_A1():
    return "SELECT COUNT(*) FROM detailedstrain WHERE identificationtechnique = 'ELISA' AND gmoconstructioninfo = 'Gene Knockout';"
def get_C2_A2():
    return "SELECT * FROM detailedstrain WHERE pathogenicity = 'Pathogen for Plants' AND typestrain = true;"
def get_C2_A3():
    return "SELECT COUNT(*) FROM detailedstrain WHERE remarks = 'FPO 8' OR accessionnumber = 'ABC123';"
def get_C2_A4():
    return "SELECT * FROM detailedstrain WHERE otherdenominations = 'MVPC1 10' AND identificationtechnique = 'PCR';"
def get_C2_A5():
    return "SELECT COUNT(*) FROM detailedstrain WHERE otherccnumbers = 'CC123' OR commentontaxonomy = 'Lactobacillus plantarum';"
def get_C2_A6():
    return "SELECT * FROM detailedstrain WHERE pathogenicity = 'Pathogen for Plants' OR gmoconstructioninfo = 'Gene Knockout';"
def get_C2_A7():
    return "SELECT COUNT(*) FROM detailedstrain WHERE name = 'Tobacco ringspot virus' AND riskgroup = 3;"
def get_C2_A8():
    return "SELECT * FROM detailedstrain WHERE name = 'Bean common mosaic virus' OR organismtype = 'Virus';"
def get_C2_A9():
    return "SELECT COUNT(*) FROM detailedstrain WHERE pathotypeserotypetype = 'diverso' AND scientificname = 'DENV-1';"
def get_C2_A10():
    return "SELECT * FROM detailedstrain WHERE name = 'Zika' AND isolationhost = 'Gesneriaceae/Sinningia/speciosa';"
################################################################################


################################################################################
# Questions and Queries of category C3: (only on detailed strain table)

# Questions:
def get_C3_Q1():
    return "Count the number of strains where identificationtechniques is 'ELISA'."
def get_C3_Q2():
    return "Select all strains where pathogenicitie is 'Pathogen for Plants'."
def get_C3_Q3():
    return "Count the number of strains where accessionnumber is 'ABC123' and gmoconstructioninfos is 'Gene Knockout'."
def get_C3_Q4():
    return "Select all strains where otherdenominationss is 'MVPC1 10'."
def get_C3_Q5():
    return "Count the number of strains where otherccnumber is 'CC123' or commentontaxonomys is 'Confirmed'."
def get_C3_Q6():
    return "Select all strains where pathogenicitie is 'Pathogen for Plants' or gmoconstructioninf is 'Gene Knockout'."
def get_C3_Q7():
    return "Count the number of strains where names is 'Influenza' and riskgroup is 3."
def get_C3_Q8():
    return "Select all strains where name is 'Ebola' or organismtype is 'Filovirus'."
def get_C3_Q9():
    return "Count the number of strains where pathotypeserotypetypes is 'Dengue' and scientificnames is 'DENV-1'."
def get_C3_Q10():
    return "Select all strains where names is 'Zika' and isolationhosts is 'Human'."


# Answers (Queries):
def get_C3_A1():
    return "SELECT COUNT(*) FROM detailedstrain WHERE identificationtechnique = 'ELISA';"
def get_C3_A2():
    return "SELECT * FROM detailedstrain WHERE pathogenicity = 'Pathogen for Plants';"
def get_C3_A3():
    return "SELECT COUNT(*) FROM detailedstrain WHERE accessionnumber = 'ABC123' AND gmoconstructioninfo = 'Gene Knockout';"
def get_C3_A4():
    return "SELECT * FROM detailedstrain WHERE otherdenominations = 'MVPC1 10';"
def get_C3_A5():
    return "SELECT COUNT(*) FROM detailedstrain WHERE otherccnumbers = 'CC123' OR commentontaxonomy = 'Confirmed';"
def get_C3_A6():
    return "SELECT * FROM detailedstrain WHERE pathogenicity = 'Pathogen for Plants' OR gmoconstructioninfo = 'Gene Knockout';"
def get_C3_A7():
    return "SELECT COUNT(*) FROM detailedstrain WHERE name = 'Influenza' AND riskgroup = 3;"
def get_C3_A8():
    return "SELECT * FROM detailedstrain WHERE name = 'Ebola' OR organismtype = 'Filovirus';"
def get_C3_A9():
    return "SELECT COUNT(*) FROM detailedstrain WHERE pathotypeserotypetype = 'Dengue' AND scientificname = 'DENV-1';"
def get_C3_A10():
    return "SELECT * FROM detailedstrain WHERE name = 'Zika' AND isolationhost = 'Human';"
################################################################################


################################################################################
# Questions and Queries of category C4: (Queries on data fields – 'Collectiondata' - detailedstrain table)

# Questions: (top 10))
def get_C4_Q1():
    return "Select all strains that have the field collectiondate with a data before '2014'."
def get_C4_Q2():
    return "Select all strains that have the field collectiondate with a data before '06-07-1995'."
def get_C4_Q3():
    return "Count all strains that have the field collectiondate with a data after '2005'."
def get_C4_Q4():
    return "Select all strains that have the field collectiondate with a data after '15-07-2010'."
def get_C4_Q5():
    return "Select all strains with collectiondate before '2000'."
def get_C4_Q6():
    return "Count all strains with collectiondate before '01-01-1995'."
def get_C4_Q7():
    return "Select all strains collected after '2015'."
def get_C4_Q8():
    return "Select all strains collected after '22-08-2002'."
def get_C4_Q9():
    return "Count all strains that have the field collectiondate with a data before '1980'."
def get_C4_Q10():
    return "Select all strains collected before '10-10-2010'."


# Questions: (10 more)
def get_C4_Q11():
    return "Select all strains collected before '2018'."
def get_C4_Q12():
    return "Select all strains with collectiondate before '10-05-2000'."
def get_C4_Q13():
    return "Count all strains collected after '2010'."
def get_C4_Q14():
    return "Select all strains collected after '12-11-2005'."
def get_C4_Q15():
    return "Select all strains collected before '1999'."
def get_C4_Q16():
    return "Count all strains with collectiondate before '30-12-1990'."
def get_C4_Q17():
    return "Select all strains collected after '2020'."
def get_C4_Q18():
    return "Select all strains with collectiondate after '01-01-2010'."
def get_C4_Q19():
    return "Count all strains collected before '1985'."
def get_C4_Q20():
    return "Select all strains with collectiondate before '15-08-2005'."


# Questions: (10 more on date ranges, eg between '1990 and 2000' or between '10-10-2005 and 10-10-2015')
def get_C4_Q21():
    return "Select all strains with collectiondate between '1980' and '1990'."
def get_C4_Q22():
    return "Select all strains collected between '12-12-2000' and '12-12-2010'."
def get_C4_Q23():
    return "Select all strains with collectiondate between '1995' and '2005'."
def get_C4_Q24():
    return "Select all strains with collectiondate between '01-01-1990' and '01-01-2000'."
def get_C4_Q25():
    return "Select all strains collected between '1985' and '1995'."
def get_C4_Q26():
    return "Select all strains collected between '10-10-2005' and '10-10-2015'."
def get_C4_Q27():
    return "Select all strains with collectiondate between '1970' and '1980'."
def get_C4_Q28():
    return "Select all strains with collectiondate between '15-03-2000' and '15-03-2010'."
def get_C4_Q29():
    return "Select all strains collected between '2000' and '2015'."
def get_C4_Q30():
    return  "Select all strains collected between '05-02-2010' and '16-04-2020'."


# Questions: (10 more queries on collection in combination with other fields)
def get_C4_Q31():
    return "Select all strains with accessionnumber '0028' and collected after '2008'. The field 'availablefordis' must be equal to true."
def get_C4_Q32():
    return "Select all strains with gmo 'true' and collected before '01-01-1990' and the identificationtechnique should be 'PCR'."
def get_C4_Q33():
    return "Select all strains with collection '5' and that have collectiondate after '2012'. The 'typestrain' should be equal to false."
def get_C4_Q34():
    return "Select all strains with identificationtechnique 'ELISA' and collectiondate before '1995'. The strain should have registeredcollection equal to true."
def get_C4_Q35():
    return "Select all strains with collection '10' and collectiondate after '10-10-2000'. The strain should have remarks containing 'Neotype'."
def get_C4_Q36():
    return "Select all strains with collection '15' and collected before '1985'. I also want the pathogenicity to be true."
def get_C4_Q37():
    return "Select all strains with identificationtechnique 'Bioassay' and with collectiondate after '2002' and the field beforenagoya must be equal to true."
def get_C4_Q38():
    return "Select all strains with collection '20' and with collectiondate before '15-08-2010'. The strain should have a commentontaxonomy."
def get_C4_Q39():
    return "Select all strains with collection '50' and collectiondate after '01-01-1995'. I also want the geoorigin to be '1046' or '1047'."
def get_C4_Q40():
    return "Select all strains with accessionnumber '09077' and typestrain equal to true and with collectiondate before '2000'."





# Answers (Queries):
def get_C4_A1():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '2014-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate < '2014');"
def get_C4_A2():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '1995-07-06') OR (LENGTH(collectiondate) = 4 AND collectiondate <= '1995');"
def get_C4_A3():
    return "SELECT COUNT(*) FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) > '2005-12-31') OR (LENGTH(collectiondate) = 4 AND collectiondate > '2005');"
def get_C4_A4():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) > '2010-07-15') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '2010');"
def get_C4_A5():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '2000-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate < '2000');"
def get_C4_A6():
    return "SELECT COUNT(*) FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '1995-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate < '1995');"
def get_C4_A7():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) > '2015-12-31') OR (LENGTH(collectiondate) = 4 AND collectiondate > '2015');"
def get_C4_A8():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) > '2002-08-22') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '2002');"
def get_C4_A9():
    return "SELECT COUNT(*) FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '1980-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate < '1980');"
def get_C4_A10():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '2010-10-10') OR (LENGTH(collectiondate) = 4 AND collectiondate <= '2010');"


def get_C4_A11():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '2018-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate < '2018');"
def get_C4_A12():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '2000-05-10') OR (LENGTH(collectiondate) = 4 AND collectiondate <= '2000');"
def get_C4_A13():
    return "SELECT COUNT(*) FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) > '2010-12-31') OR (LENGTH(collectiondate) = 4 AND collectiondate > '2010');"
def get_C4_A14():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) > '2005-11-12') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '2005');"
def get_C4_A15():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '1999-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate < '1999');"
def get_C4_A16():
    return "SELECT COUNT(*) FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '1990-12-30') OR (LENGTH(collectiondate) = 4 AND collectiondate <= '1990');"
def get_C4_A17():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) > '2020-12-31') OR (LENGTH(collectiondate) = 4 AND collectiondate > '2020');"
def get_C4_A18():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) > '2010-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '2010');"
def get_C4_A19():
    return "SELECT COUNT(*) FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '1985-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate < '1985');"
def get_C4_A20():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '2005-08-15') OR (LENGTH(collectiondate) = 4 AND collectiondate <= '2005');"



def get_C4_A21():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) >= '1980-01-01' AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) <= '1990-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '1980' AND collectiondate <= '1990');"
def get_C4_A22():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) >= '2000-12-12' AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) <= '2010-12-12') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '2000' AND collectiondate <= '2010');"
def get_C4_A23():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) >= '1995-01-01' AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) <= '2005-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '1995' AND collectiondate <= '2005');"
def get_C4_A24():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) >= '1990-01-01' AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) <= '2000-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '1990' AND collectiondate <= '2000');"
def get_C4_A25():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) >= '1985-01-01' AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) <= '1995-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '1985' AND collectiondate <= '1995');"
def get_C4_A26():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) >= '2005-10-10' AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) <= '2015-10-10') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '2005' AND collectiondate <= '2015');"
def get_C4_A27():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) >= '1970-01-01' AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) <= '1980-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '1970' AND collectiondate <= '1980');"
def get_C4_A28():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) >= '2000-03-15' AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) <= '2010-03-15') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '2000' AND collectiondate <= '2010');"
def get_C4_A29():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) >= '2000-01-01' AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) <= '2015-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '2000' AND collectiondate <= '2015');"
def get_C4_A30():
    return "SELECT * FROM detailedstrain WHERE (LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) >= '2010-02-05' AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) <= '2020-04-16') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '2010' AND collectiondate <= '2020');"


def get_C4_A31():
    return "SELECT * FROM detailedstrain WHERE accessionnumber = '0028' AND ((LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) > '2008-12-31') OR (LENGTH(collectiondate) = 4 AND collectiondate > '2008')) AND availablefordis = true;"
def get_C4_A32():
    return "SELECT * FROM detailedstrain WHERE gmo = true AND ((LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '1990-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate < '1990')) AND identificationtechnique = 'PCR';"
def get_C4_A33():
    return "SELECT * FROM detailedstrain WHERE collection = '5' AND ((LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) > '2012-12-31') OR (LENGTH(collectiondate) = 4 AND collectiondate > '2012')) AND typestrain = false;"
def get_C4_A34():
    return "SELECT * FROM detailedstrain WHERE identificationtechnique = 'ELISA' AND ((LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '1995-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate < '1995')) AND registeredcollection = true;"
def get_C4_A35():
    return "SELECT * FROM detailedstrain WHERE collection = '10' AND ((LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) > '2000-10-10') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '2000')) AND remarks LIKE '%Neotype%';"
def get_C4_A36():
    return "SELECT * FROM detailedstrain WHERE collection = '15' AND ((LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '1985-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate < '1985')) AND pathogenicity = true;"
def get_C4_A37():
    return "SELECT * FROM detailedstrain WHERE identificationtechnique = 'Bioassay' AND ((LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) > '2002-12-31') OR (LENGTH(collectiondate) = 4 AND collectiondate > '2002')) AND nagoyaconditions = true;"
def get_C4_A38():
    return "SELECT * FROM detailedstrain WHERE collection = '20' AND ((LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '2010-08-15') OR (LENGTH(collectiondate) = 4 AND collectiondate <= '2010')) AND commentontaxonomy IS NOT NULL;"
def get_C4_A39():
    return "SELECT * FROM detailedstrain WHERE collection = '50' AND ((LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) > '1995-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate >= '1995')) AND (geoorigin = '1046' OR geoorigin = '1047');"
def get_C4_A40():
    return "SELECT * FROM detailedstrain WHERE accessionnumber = '09077' AND typestrain = true AND ((LENGTH(collectiondate) = 10 AND SUBSTR(collectiondate, 7, 4) || '-' || SUBSTR(collectiondate, 4, 2) || '-' || SUBSTR(collectiondate, 1, 2) < '2000-01-01') OR (LENGTH(collectiondate) = 4 AND collectiondate < '2000'));"
################################################################################



################################################################################
# Questions and Queries of the C5 category: (Queries with GROUP BY) – (only on detailedstrain table).

# Questions:
def get_C5_Q1():
    return "Select and count the number of strains from the strain table, grouping the results by the identificationtechnique field."

def get_C5_Q2():
    return "Select and count all the strains that have the 'pathogenicity' field equal to 'Plant' and group only those that have the same collectiondate. I want you to also show me the collectiondate field."

def get_C5_Q3():
    return "Select and count the number of strains by grouping them based on the gmo and collection fields."

def get_C5_Q4():
    return "Select and count all strains that have the 'typestrain' field equal to true and group only those that have the same inclusiondate. I want you to also show me the inclusiondate field and only the groups that have more than 3 examples."

def get_C5_Q5():
    return "Select and count the number of strains from the strain table, grouping the results by the scientificname field."

def get_C5_Q6():
    return "Select and count all the strains that have the 'veuquarantine' field equal to true and group only those that have the same pathotypeserotypetype. I want you to also show me the pathotypeserotypetype field."

def get_C5_Q7():
    return "Select and count the number of strains by grouping them based on the riskgroup and lyticcicle fields."

def get_C5_Q8():
    return "Select and count all strains that have the 'ineva' field equal to true and group only those that have the same propagationhost. I want you to also show me the propagationhost field and only the groups that have more than 2 examples."

def get_C5_Q9():
    return "Select and count the number of strains from the strain table, grouping the results by the remarks field."

def get_C5_Q10():
    return "Select and count all the strains that have the 'contamination' field equal to true and group only those that have the same symptomatologyonisolationhost. I want you to also show me the symptomatologyonisolationhost field."


# Answers (Queries):
def get_C5_A1():
    return "SELECT COUNT(*), identificationtechnique FROM detailedstrain GROUP BY identificationtechnique;"

def get_C5_A2():
    return "SELECT COUNT(*), collectiondate FROM detailedstrain WHERE pathogenicity = 'Plant' GROUP BY collectiondate;"

def get_C5_A3():
    return "SELECT COUNT(*), gmo, collection FROM detailedstrain GROUP BY gmo, collection;"

def get_C5_A4():
    return "SELECT inclusiondate, COUNT(*) FROM detailedstrain WHERE typestrain = TRUE GROUP BY inclusiondate HAVING COUNT(*) > 3;"

def get_C5_A5():
    return "SELECT COUNT(*), scientificname FROM detailedstrain GROUP BY scientificname;"

def get_C5_A6():
    return "SELECT COUNT(*), pathotypeserotypetype FROM detailedstrain WHERE veuquarantine = TRUE GROUP BY pathotypeserotypetype;"

def get_C5_A7():
    return "SELECT COUNT(*), riskgroup, lyticcicle FROM detailedstrain GROUP BY riskgroup, lyticcicle;"

def get_C5_A8():
    return "SELECT propagationhost, COUNT(*) FROM detailedstrain WHERE ineva = TRUE GROUP BY propagationhost HAVING COUNT(*) > 2;"

def get_C5_A9():
    return "SELECT COUNT(*), remarks FROM detailedstrain GROUP BY remarks;"

def get_C5_A10():
    return "SELECT COUNT(*), symptomatologyonisolationhost FROM detailedstrain WHERE contamination = TRUE GROUP BY symptomatologyonisolationhost;"

################################################################################################################################################################

