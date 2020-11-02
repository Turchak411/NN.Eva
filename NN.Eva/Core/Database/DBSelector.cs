using System;
using System.Collections.Generic;
using MySql.Data.MySqlClient;
using NN.Eva.Models;
using NN.Eva.Models.Database;
using NN.Eva.Services;
using NN.Eva.Services.Database;

namespace NN.Eva.Core.Database
{
    public class DBSelector
    {
        /// <summary>
        /// Database configuarion
        /// </summary>
        public DatabaseConfig DatabaseConfiguration { get; set; }

        public DBSelector(DatabaseConfig databaseConfiguration)
        {
            DatabaseConfiguration = databaseConfiguration;
        }

        public List<Guid> GetLayerIDList(Guid networkID)
        {
            string query = "SELECT * FROM " + DatabaseConfiguration.Database + ".layers WHERE networkId = '" + networkID + "'  ORDER BY number";

            MySqlConnection connection = GetNewDBConnection();

            List<Guid> guidList = new List<Guid>();

            using (connection)
            {
                connection.Open();

                var command = new MySqlCommand(query, connection);

                using (MySqlDataReader reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        Guid guid = Guid.Parse(reader[0].ToString());

                        guidList.Add(guid);
                    }
                }

                connection.Close();
            }

            return guidList;
        }

        public List<Guid> GetNeuronsIDList(Guid layerID)
        {
            string query = "SELECT * FROM " + DatabaseConfiguration.Database + ".neurons WHERE networkId = '" + layerID + "' ORDER BY numberInLayer";

            MySqlConnection connection = GetNewDBConnection();

            List<Guid> guidList = new List<Guid>();

            using (connection)
            {
                connection.Open();

                var command = new MySqlCommand(query, connection);

                using (MySqlDataReader reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        Guid guid = Guid.Parse(reader[0].ToString());

                        guidList.Add(guid);
                    }
                }

                connection.Close();
            }

            return guidList;
        }

        public List<double> GetWeightsList(Guid neuronID)
        {
            string query = "SELECT * FROM " + DatabaseConfiguration.Database + ".neurons WHERE networkId = '" + neuronID + "' ORDER BY numberInNeuron";

            MySqlConnection connection = GetNewDBConnection();

            List<double> weightList = new List<double>();

            using (connection)
            {
                connection.Open();

                var command = new MySqlCommand(query, connection);

                using (MySqlDataReader reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        double weight = double.Parse(reader[0].ToString());

                        weightList.Add(weight);
                    }
                }

                connection.Close();
            }

            return weightList;
        }

        private MySqlConnection GetNewDBConnection()
        {
            MySqlConnection connection = null;

            try
            {
                var dbConfigurator = new DatabaseConfigurator();

                connection = new MySqlConnection(dbConfigurator.ReturnDatabaseConnection(DatabaseConfiguration));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{DateTime.Now}\nException: {ex}");
                Logger.LogError(ErrorType.DBConnectionError, ex);
            }

            return connection;
        }
    }
}
