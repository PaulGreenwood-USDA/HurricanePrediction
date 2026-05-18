// Subscription-scoped entry for azd. Creates an RG and deploys app resources.
targetScope = 'subscription'

@minLength(1)
@description('Name of the azd environment; used to derive unique resource names.')
param environmentName string

@minLength(1)
@description('Primary location for all resources.')
param location string

@description('SKU for the App Service Plan.')
@allowed([
  'B1'
  'B2'
  'P0v3'
  'P1v3'
])
param appServicePlanSku string = 'B1'

var resourceToken = uniqueString(subscription().id, environmentName, location)
var tags = {
  'azd-env-name': environmentName
}

resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: 'rg-${environmentName}'
  location: location
  tags: tags
}

module resources 'resources.bicep' = {
  name: 'resources'
  scope: rg
  params: {
    location: location
    resourceToken: resourceToken
    appServicePlanSku: appServicePlanSku
    tags: tags
  }
}

output AZURE_LOCATION string = location
output AZURE_RESOURCE_GROUP string = rg.name
output WEB_APP_NAME string = resources.outputs.WEB_APP_NAME
output WEB_APP_URI string = resources.outputs.WEB_APP_URI
output APPLICATIONINSIGHTS_CONNECTION_STRING string = resources.outputs.APPLICATIONINSIGHTS_CONNECTION_STRING
