using System;
using MySql.Data.MySqlClient;
using NN.Eva.Models;
using NN.Eva.Services;
using NN.Eva.Models.Database;
using NN.Eva.Services.Database;

namespace NN.Eva.Core.Database
{
    public class DBDeleter
    {
        /// <summary>
        /// Database configuarion
        /// </summary>
        public DatabaseConfig DatabaseConfiguration { get; set; }

        private Logger _logger;

        public DBDeleter(Logger logger, DatabaseConfig databaseConfiguration)
        {
            _logger = logger;

            DatabaseConfiguration = databaseConfiguration;
        }

        public void DeleteFromTableNetworks(Guid id)
        {
            DeleteFromTable(id, "networks", "id");
        }

        public void DeleteFromTableLayers(Guid networkId)
        {
            DeleteFromTable(networkId, "layers", "networkId");
        }

        public void DeleteFromTableNeurons(Guid layerId)
        {
            DeleteFromTable(layerId, "neurons", "layerId");
        }

        public void DeleteFromTableWeights(Guid neuronId)
        {
            DeleteFromTable(neuronId, "weights", "neuronId");
        }

        private void DeleteFromTable(Guid id, string tableName, string rowIdName)
        {
            try
            {
                string query = "DELETE FROM " + DatabaseConfiguration.Database + "." + tableName + " WHERE (`" + rowIdName + "` = '" + id + "');";

                MySqlConnection connection = GetNewDBConnection();

                var command = new MySqlCommand(query, connection);
                command.ExecuteNonQuery();
            }
            catch (Exception ex)
            {
                _logger.LogError(ErrorType.DBDeleteError, "Delete-error in table \"" + tableName + "\"!\nID: " + id + "\n" + ex);
            }
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
                Console.WriteLine($"{DateTime.Now }\nException: {ex}");
                _logger.LogError(ErrorType.DBConnectionError, ex);
            }

            return connection;
        }
    }
}
