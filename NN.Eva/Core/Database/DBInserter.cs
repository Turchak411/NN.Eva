using System;
using System.Collections.Generic;
using System.Text;

namespace NN.Eva.Core.Database
{
    public class DBInserter
    {
        private MySqlConnection _connection;

        private Logger _logger;

        public DBInserter(MySqlConnection connection, Logger logger)
        {
            _connection = connection;
            _logger = logger;
        }

        public void InsertNetwork(Guid id, Guid userId, int iterations, string networkStructure)
        {
            try
            {
                string savingDate = DateTime.Now.Year + "-" + DateTime.Now.Month + "-" + DateTime.Now.Day + " " +
                                   +DateTime.Now.Hour + ":" + DateTime.Now.Minute + ":" + DateTime.Now.Second;

                string query =
                    $"INSERT INTO memorynn.networks VALUES(\'{id}\',\'{userId}\',\'{savingDate}\'," +
                    $"\'{networkStructure}\',\'{iterations}\');";

                var command = new MySqlCommand(query, _connection);
                command.ExecuteNonQuery();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Insert-error in table \"networks\"!");
                _logger.LogError(ErrorType.DBInsertError, "Table 'networks'.\n" + ex);
            }
        }

        public void InsertLayer(Guid id, Guid networkId, int number)
        {
            try
            {
                string query = $"INSERT INTO memorynn.layers VALUES(\'{id}\',\'{networkId}\','{number}');";

                var command = new MySqlCommand(query, _connection);
                command.ExecuteNonQuery();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Insert-error in table \"layers\"!");
                _logger.LogError(ErrorType.DBInsertError, "Table 'networks'.\n" + ex);
            }
        }

        public void InsertNeuron(Guid id, Guid layerId, int number)
        {
            try
            {
                string query = $"INSERT INTO memorynn.neurons VALUES(\'{id}\',\'{layerId}\','{number}');";

                var command = new MySqlCommand(query, _connection);
                command.ExecuteNonQuery();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Insert-error in table \"neurons\"!");
                _logger.LogError(ErrorType.DBInsertError, "Table 'networks'.\n" + ex);
            }
        }

        public void InsertWeight(Guid neuronId, int number, double value)
        {
            try
            {
                string query = $"INSERT INTO memorynn.weights VALUES(\'{Guid.NewGuid()}\',\'{neuronId}\',\'{number}\',\'{value.ToString().Replace(',', '.')}\');";

                var command = new MySqlCommand(query, _connection);
                command.ExecuteNonQuery();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Insert-error in table \"weights\"!");
                _logger.LogError(ErrorType.DBInsertError, "Table 'networks'.\n" + ex);
            }
        }
    }
}
