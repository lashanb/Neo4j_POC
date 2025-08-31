import csv

def split_contracts():
    contracts_file_path = "C:\\Users\\lasha\\Dropbox\\My Documents\\Ontology\\Neo4j PoC\\VS Code\\Sample Data\\contracts.csv"
    collateral_file_path = "C:\\Users\\lasha\\Dropbox\\My Documents\\Ontology\\Neo4j PoC\\VS Code\\Sample Data\\collateral.csv"
    new_contracts_file_path = "C:\\Users\\lasha\\Dropbox\\My Documents\\Ontology\\Neo4j PoC\\VS Code\\Sample Data\\contracts_new.csv"
    linking_file_path = "C:\\Users\\lasha\\Dropbox\\My Documents\\Ontology\\Neo4j PoC\\VS Code\\Sample Data\\contract_collateral.csv"

    collateral_rows = []
    new_contracts_rows = []
    linking_rows = []

    with open(contracts_file_path, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        
        # Separate headers
        collateral_headers = reader.fieldnames
        new_contracts_headers = reader.fieldnames
        linking_headers = ["contractId", "collateralId"]

        for row in reader:
            if row['contractType'] == 'Collateral':
                collateral_rows.append(row)
            else:
                new_contracts_rows.append(row)

    # Create linking data (assuming 1-to-1 mapping for now)
    for i in range(min(len(new_contracts_rows), len(collateral_rows))):
        linking_rows.append({
            "contractId": new_contracts_rows[i]["contractId"],
            "collateralId": collateral_rows[i]["contractId"]
        })

    # Write collateral data
    with open(collateral_file_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=collateral_headers)
        writer.writeheader()
        writer.writerows(collateral_rows)

    # Write new contracts data
    with open(new_contracts_file_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=new_contracts_headers)
        writer.writeheader()
        writer.writerows(new_contracts_rows)

    # Write linking data
    with open(linking_file_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=linking_headers)
        writer.writeheader()
        writer.writerows(linking_rows)

if __name__ == "__main__":
    split_contracts()
