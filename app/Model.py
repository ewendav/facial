import sqlite3
import requests
import json

class Model:
    def __init__(self):
        self.connection = sqlite3.connect('../database/ewenLogin.db')

        cursor = self.connection.cursor()
        insert_query = """
        CREATE TABLE IF NOT EXISTS log (
        id INTEGER PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        numero_etape INTEGER,
        login TEXT,
        commentaire TEXT,
        numero_badge TEXT
        );
        """
        cursor.execute(insert_query)
        self.connection.commit()
        cursor.close()



    def insertLog(self,numEtape, login, commentaire, numBadge):
        cursor = self.connection.cursor()

        insert_query = """
        INSERT INTO log (numero_etape, login, commentaire, numero_badge)
        VALUES (?, ?, ?, ?)
        """
        values_to_insert = (numEtape, login, commentaire, numBadge)
        cursor.execute(insert_query, values_to_insert)

        self.connection.commit()
        cursor.close()


    def check_credentials(self, username, password ):
        vRetour = False
        
        response = requests.get(f"https://www.btssio-carcouet.fr/ppe4/public/connect2/{username}/{password}/infirmiere")
        responseText = response.text
        responseText = responseText.replace("'", "\"")

        try : 
            if response.status_code == 200:

                dataResponse = json.loads(responseText)

                if dataResponse.get('id', False) :
                    self.insertLog(1,username, 'authentification succes', None)
                    vRetour = True
                else : 
                    self.insertLog(1,username, 'authentification failled', None)
            else : 
                self.insertLog(1,username, 'error code response : ' + response.status_code, None)
            
        except Exception as e:
            self.insertLog(1,username, 'error exception', None)

        return vRetour
    

    def check_badge(self, idBadge, username):
        vRetour = False
                print('oe')

        
        response = requests.get(f'https://www.btssio-carcouet.fr/ppe4/public/badge/{username}/{str(id)}')
        responseText = response.text
        responseText = responseText.replace("'", "\"")

        try : 
            if response.status_code == 200:

                dataResponse = json.loads(responseText)

                print(dataResponse)

                if dataResponse.get('id', False) :
                    self.insertLog(2,username, 'lecture badge succes', None)
                    vRetour = True
                else : 
                    self.insertLog(2,username, 'lecture badge failled', None)
            else : 
                self.insertLog(2,username, 'error code response : ' + response.status_code, None)
            
        except Exception as e:
            self.insertLog(1,username, 'error exception', None)

        return vRetour


