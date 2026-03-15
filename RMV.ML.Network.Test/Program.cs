//using NLog;
using System.Diagnostics;

using RMV.ML.Network.Common;
using RMV.ML.Network.Domain;

namespace RMV.ML.Network.Test;

/// <summary>
/// Neural Network Test Application
/// </summary>
class Program
{
	//static Logger logger = LogManager.GetCurrentClassLogger();	

	/// <summary>
	/// Main entry point
	/// </summary>	
	static void Main()
	{
		var timer = new Stopwatch();
		timer.Start();

		var settings = ConfigManager.GetRoot<AppSettings>() ?? throw new Exception( "Failed to load configuration" );					

		var parser = new Parser( settings.Output );
		var trainSet = parser.Run( File.ReadAllLines( settings.Train ) );
		var testSet = parser.Run( File.ReadAllLines( settings.Test ) );

		var network = new Net( settings ) { Learning = LearningType.Batch };
		network.Connect();
		network.Initialize();
		
		string message = $"Hidden:{settings.Hidden.Join()} Iterations:{settings.Iterations} Rate:{settings.Rate} {timer.Elapsed.TotalSeconds:f2} sec";
		//logger.Info( message );
		Console.WriteLine( message );		

		Train( network, settings, trainSet, testSet, timer );		              
		
		//File.WriteAllText( settings.Errors, errCsv.Join() );
		Console.ReadLine();
	}


	/// <summary>
	/// Train network
	/// </summary>
	/// <param name="network">network</param>
	/// <param name="settings">application settings</param>
	/// <param name="timer">stopwatch</param>
	static void Train( Net network, AppSettings settings, DataSet trainSet, DataSet testSet, Stopwatch timer )
	{			
		for( int epoch = 0; epoch < settings.Iterations; epoch++ )
		{
			double batchErrorSum = 0;

			for( int i = 0; i < settings.Batch; i++ )
			{
				network.Training( trainSet );

				if( network.Learning == LearningType.Online ) network.Update( 1 );

				batchErrorSum += network.Error;
			}

			if( network.Learning == LearningType.Batch ) network.Update( settings.Batch );

			double averageError = batchErrorSum / settings.Batch;
						
			if( epoch % settings.Print != 0 ) // Don't test every epoch, just display error
			{
				Console.WriteLine( $"Epoch={epoch}, Avg Error={averageError:f4} Time={timer.Elapsed:hh\\:mm\\:ss}" );
				continue;
			}

			var errors = network.Testing( testSet );
			int count = errors.Count;			
			int total = testSet.Target.Count;

			string info = $"Epoch={epoch}, Avg Error={averageError:f4} Total={total} Fails={count} PC={Tools.Percent( count, total ):f2}% Time={timer.Elapsed:hh\\:mm\\:ss}";
			Console.WriteLine( info );
			Console.WriteLine();			
		}
	}
	
}
