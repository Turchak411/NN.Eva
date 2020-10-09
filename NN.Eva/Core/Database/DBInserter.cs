using System;
using MySql.Data.MySqlClient;
using NN.Eva.Models;
using NN.Eva.Models.Database;
using NN.Eva.Services;
using NN.Eva.Services.Database;

namespace NN.Eva.Core.Database
{
    public class DBInserter
    {
        /// <summary>
        /// Database configuarion
        /// </summary>
        public DatabaseConfig DatabaseConfiguration { get; set; }

        private Logger _logger;

        public DBInserter(Logger logger, DatabaseConfig databaseConfiguration)
        {
            _logger = logger;

            DatabaseConfiguration = databaseConfiguration;
        }

        /// <summary>
        /// Inserting networks data to table "networks"
        /// </summary>
        /// <param name="id"></param>
        /// <param name="userId"></param>
        /// <param name="iterations"></param>
        /// <param name="networkStructure"></param>
        public void InsertNetwork(Guid id, Guid userId, int iterations, string networkStructure)
        {
            string savingDate = DateTime.Now.Year + "-" + DateTime.Now.Month + "-" + DateTime.Now.Day + " " +
                                        +DateTime.Now.Hour + ":" + DateTime.Now.Minute + ":" + DateTime.Now.Second;

            string query =
                        $"INSERT INTO {DatabaseConfiguration.Database}.networks VALUES(\'{id}\',\'{userId}\',\'{savingDate}\'," +
                        $"\'{networkStructure}\',\'{iterations}\');";

            InsertInTable(query, "networks");
        }

        /// <summary>
        /// Inserting layers data to table "layers"
        /// </summary>
        /// <param name="id"></param>
        /// <param name="userId"></param>
        /// <param name="networkId"></param>
        /// <param name="number"></param>
        public void InsertLayer(Guid id, Guid userId, Guid networkId, int number)
        {
            string query =
                $"INSERT INTO {DatabaseConfiguration.Database}.layers " +
                $"VALUES(\'{id}\',\'{userId}\',\'{networkId}\','{number}');";

            InsertInTable(query, "layers");
        }

        /// <summary>
        /// Inserting neurons data to table "neurons"
        /// </summary>
        /// <param name="id"></param>
        /// <param name="userId"></param>
        /// <param name="layerId"></param>
        /// <param name="number"></param>
        /// <param name="offsetValue"></param>
        /// <param name="offsetWeight"></param>
        public void InsertNeuron(Guid id, Guid userId, Guid layerId, int number, double offsetValue, double offsetWeight)
        {
            string query = $"INSERT INTO {DatabaseConfiguration.Database}.neurons " +
                           $"VALUES(\'{id}\',\'{userId}\',\'{layerId}\','{number}'," +
                           $"'{offsetValue.ToString().Replace(',', '.')}','{offsetWeight.ToString().Replace(',', '.')}');";

            InsertInTable(query, "neurons");
        }

        /// <summary>
        /// Inserting weights data to table "weights"
        /// </summary>
        /// <param name="neuronId"></param>
        /// <param name="userId"></param>
        /// <param name="number"></param>
        /// <param name="value"></param>
        public void InsertWeight(Guid neuronId, Guid userId, int number, double value)
        {
            string query = $"INSERT INTO {DatabaseConfiguration.Database}.weights " +
                           $"VALUES(\'{Guid.NewGuid()}\',\'{userId}\'," +
                           $"\'{neuronId}\',\'{number}\',\'{value.ToString().Replace(',', '.')}\');";

            InsertInTable(query, "weights");        
        }

        /// <summary>
        /// Executing insert-query
        /// </summary>
        /// <param name="query"></param>
        /// <param name="tableName"></param>
        private void InsertInTable(string query, string tableName)
        {
            // Connection to Database:
            MySqlConnection connection = null;

            try
            {
                connection = GetNewDBConnection();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error in connection to Database!");
                _logger.LogError(ErrorType.DBConnectionError, ex);
            }

            try
            {
                using (connection)
                {
                    connection.Open();
                    
                    var command = new MySqlCommand(query, connection);
                    command.ExecuteNonQuery();

                    connection.Close();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Insert-error in table \"{tableName}\"!");
                _logger.LogError(ErrorType.DBInsertError, $"Table '{tableName}'.\n" + ex);
            }
        }

        /// <summary>
        /// Getting new DB connect
        /// </summary>
        /// <returns></returns>
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
