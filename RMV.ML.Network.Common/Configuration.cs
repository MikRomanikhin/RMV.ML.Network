using Microsoft.Extensions.Configuration;

namespace RMV.ML.Network.Configuration;

/// <summary>
/// Custom configuration section handler
/// </summary>
public static class ConfigManager
{

	#region Initialise ---------------------------------------------------------

	/// <summary>
	/// The root of the configuration
	/// </summary>
	static readonly IConfigurationRoot root = GetConfigurationRoot( AppDomain.CurrentDomain.BaseDirectory );
		
	/// <summary>
	/// Gets the configuration root
	/// </summary>
	/// <param name="path">The base path for the configuration files</param>
	/// <returns>The configuration root</returns>
	static IConfigurationRoot GetConfigurationRoot( string path )
	{
		var environment = Environment.GetEnvironmentVariable( "ASPNETCORE_ENVIRONMENT" ) ?? "Production";

		return new ConfigurationBuilder().SetBasePath( path )			
			.AddJsonFile( "appSettings.json", optional: true, reloadOnChange: true ) 
			// support environment-specific overrides (e.g., appSettings.Development.json)
			.AddJsonFile( $"appSettings.{environment}.json", optional: true, reloadOnChange: true )
			.Build();
	}

	#endregion


	/// <summary>
	/// Binds the root of the configuration directly to a type
	/// </summary>
	public static T? GetRoot<T>() => root.Get<T>();

	/// <summary>
	/// Retrieves custom configuration section by type or name 
	/// </summary>
	/// <typeparam name="T">section type</typeparam>
	/// <param name="name">section name (optional)</param>
	/// <returns>configuration section</returns>
	public static T? GetSection<T>( string? name = null ) => root.GetSection( name ?? typeof( T ).Name ).Get<T>();

	/// <summary>
	/// Retrieves single value
	/// </summary>
	/// <typeparam name="T">value type</typeparam>
	/// <param name="name">value name</param>
	/// <returns>value</returns>
	public static T? GetValue<T>( string name ) => root.GetValue<T>( name );
}

