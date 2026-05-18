// App-level resources: Log Analytics, App Insights, App Service Plan, Web App.
targetScope = 'resourceGroup'

param location string
param resourceToken string
param appServicePlanSku string
param tags object

var abbrs = {
  appServicePlan: 'plan-'
  webApp: 'app-'
  logAnalytics: 'log-'
  appInsights: 'appi-'
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${abbrs.logAnalytics}${resourceToken}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${abbrs.appInsights}${resourceToken}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

resource plan 'Microsoft.Web/serverfarms@2024-04-01' = {
  name: '${abbrs.appServicePlan}${resourceToken}'
  location: location
  tags: tags
  sku: { name: appServicePlanSku }
  kind: 'linux'
  properties: { reserved: true }
}

resource web 'Microsoft.Web/sites@2024-04-01' = {
  name: '${abbrs.webApp}${resourceToken}'
  location: location
  // azd matches services by this tag value to the `services.web` entry in azure.yaml
  tags: union(tags, { 'azd-service-name': 'web' })
  kind: 'app,linux'
  identity: { type: 'SystemAssigned' }
  properties: {
    serverFarmId: plan.id
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.12'
      alwaysOn: appServicePlanSku != 'B1'
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
      http20Enabled: true
      appCommandLine: 'gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 2 wsgi:app'
      appSettings: [
        { name: 'SCM_DO_BUILD_DURING_DEPLOYMENT', value: 'true' }
        { name: 'ENABLE_ORYX_BUILD', value: 'true' }
        { name: 'WEBSITES_PORT', value: '8000' }
        { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
        { name: 'PYTHONUNBUFFERED', value: '1' }
      ]
    }
  }
}

output WEB_APP_NAME string = web.name
output WEB_APP_URI string = 'https://${web.properties.defaultHostName}'
output APPLICATIONINSIGHTS_CONNECTION_STRING string = appInsights.properties.ConnectionString
