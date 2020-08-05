using MySql.Data.MySqlClient;
using NN.Eva.Models;
using NN.Eva.Services;
using System;

namespace NN.Eva.Core.Database
{
    public class DBDeleter
    {
        private MySqlConnection _connection;

        private Logger _logger;

        public DBDeleter(MySqlConnection connection, Logger logger)
        {
            _connection = connection;
            _logger = logger;
        }

        public void DeleteFromTableNetworks(Guid id)
        {
            DeleteFromTable(id, "memorynn", "networks", "id");
        }

        public void DeleteFromTableLayers(Guid networkId)
        {
            DeleteFromTable(networkId, "memorynn", "layers", "networkId");
        }

        public void DeleteFromTableNeurons(Guid layerId)
        {
            DeleteFromTable(layerId, "memorynn", "neurons", "layerId");
        }

        public void DeleteFromTableWeights(Guid neuronId)
        {
            DeleteFromTable(neuronId, "memorynn", "weights", "neuronId");
        }

        private void DeleteFromTable(Guid id, string dbName, string tableName, string rowIdName)
        {
            try
            {
                string query = "DELETE FROM `" + dbName + "`.`" + tableName + "` WHERE (`" + rowIdName + "` = '" + id + "');";

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
