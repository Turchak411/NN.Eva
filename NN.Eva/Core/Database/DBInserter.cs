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

        public void InsertNetwork(Guid id, Guid userId, int iterations, string networkStructure)
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

                    string savingDate = DateTime.Now.Year + "-" + DateTime.Now.Month + "-" + DateTime.Now.Day + " " +
                                        +DateTime.Now.Hour + ":" + DateTime.Now.Minute + ":" + DateTime.Now.Second;

                    string query =
                        $"INSERT INTO {DatabaseConfiguration.Database}.networks VALUES(\'{id}\',\'{userId}\',\'{savingDate}\'," +
                        $"\'{networkStructure}\',\'{iterations}\');";

                    var command = new MySqlCommand(query, connection);
                    command.ExecuteNonQuery();

                    connection.Close();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Insert-error in table \"networks\"!");
                _logger.LogError(ErrorType.DBInsertError, "Table 'networks'.\n" + ex);
            }
        }

        public void InsertLayer(Guid id, Guid userId, Guid networkId, int number)
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
                    string query =
                        $"INSERT INTO {DatabaseConfiguration.Database}.layers VALUES(\'{id}\',\'{userId}\',\'{networkId}\','{number}');";

                    var command = new MySqlCommand(query, connection);
                    command.ExecuteNonQuery();

                    connection.Close();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Insert-error in table \"layers\"!");
                _logger.LogError(ErrorType.DBInsertError, "Table 'networks'.\n" + ex);
            }
        }

        public void InsertNeuron(Guid id, Guid userId, Guid layerId, int number)
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

                    string query =
                        $"INSERT INTO {DatabaseConfiguration.Database}.neurons VALUES(\'{id}\',\'{userId}\',\'{layerId}\','{number}');";

                    var command = new MySqlCommand(query, connection);
                    command.ExecuteNonQuery();

                    connection.Close();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Insert-error in table \"neurons\"!");
                _logger.LogError(ErrorType.DBInsertError, "Table 'networks'.\n" + ex);
            }
        }

        public void InsertWeight(Guid neuronId, Guid userId, int number, double value)
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

                    string query =
                        $"INSERT INTO {DatabaseConfiguration.Database}.weights VALUES(\'{Guid.NewGuid()}\',\'{userId}\',\'{neuronId}\',\'{number}\',\'{value.ToString().Replace(',', '.')}\');";

                    var command = new MySqlCommand(query, connection);
                    command.ExecuteNonQuery();

                    connection.Close();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Insert-error in table \"weights\"!");
                _logger.LogError(ErrorType.DBInsertError, "Table 'networks'.\n" + ex);
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
