using MySql.Data.MySqlClient;
using NN.Eva.Models;
using NN.Eva.Services;
using System;

namespace NN.Eva.Core.Database
{
    public class DBDeleter
    {
        /// <summary>
        /// Name of the database for memory
        /// </summary>
        public string DatabaseName { get; set; }

        private MySqlConnection _connection;

        private Logger _logger;

        public DBDeleter(MySqlConnection connection, Logger logger, string databaseName = "memorynn")
        {
            _connection = connection;
            _logger = logger;

            DatabaseName = databaseName;
        }

        public void DeleteFromTableNetworks(Guid id)
        {
            DeleteFromTable(id, DatabaseName, "networks", "id");
        }

        public void DeleteFromTableLayers(Guid networkId)
        {
            DeleteFromTable(networkId, DatabaseName, "layers", "networkId");
        }

        public void DeleteFromTableNeurons(Guid layerId)
        {
            DeleteFromTable(layerId, DatabaseName, "neurons", "layerId");
        }

        public void DeleteFromTableWeights(Guid neuronId)
        {
            DeleteFromTable(neuronId, DatabaseName, "weights", "neuronId");
        }

        private void DeleteFromTable(Guid id, string dbName, string tableName, string rowIdName)
        {
            try
            {
                string query = "DELETE FROM " + dbName + "." + tableName + " WHERE (`" + rowIdName + "` = '" + id + "');";

                var command = new MySqlCommand(query, _connection);
                command.ExecuteNonQuery();
            }
            catch (Exception ex)
            {
                _logger.LogError(ErrorType.DBDeleteError, "Delete-error in table \"" + tableName + "\"!\nID: " + id + "\n" + ex);
            }
        }
    }
}
