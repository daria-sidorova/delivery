import torch
import torch.nn as nn
from django.utils import timezone
from datetime import timedelta
from .models import Order
from django.db.models import Sum

class DemandPredictor(nn.Module):
    def __init__(self):
        super(DemandPredictor, self).__init__()
        self.fc1 = nn.Linear(14, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model():
    today = timezone.now().date()
    historical_data = []
    for i in range(29, -1, -1):
        day = today - timedelta(days=i)
        total = Order.objects.filter(timestamp__date=day).aggregate(total=Sum('total_price'))['total'] or 0
        count = Order.objects.filter(timestamp__date=day).count()
        historical_data.append([float(total), float(count)])

    X, y = [], []
    for i in range(len(historical_data) - 7):
        window = [item for sublist in historical_data[i:i+7] for item in sublist]
        X.append(window)
        y.append(historical_data[i+7][0])

    if not X:
        print("Not enough data for training")
        model = DemandPredictor()
        torch.save(model.state_dict(), 'demand_model.pth')
        return model

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    X_mean, X_std = X.mean(), X.std()
    y_mean, y_std = y.mean().item(), y.std().item()

    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_std

    model = DemandPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    torch.save({'state_dict': model.state_dict(), 'mean': y_mean, 'std': y_std}, 'demand_model.pth')
    print("Model trained and saved to 'demand_model.pth'")
    return model

def predict_demand(historical_data):
    model = DemandPredictor()
    try:
        checkpoint = torch.load('demand_model.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        y_mean, y_std = checkpoint['mean'], checkpoint['std']
        if isinstance(y_mean, torch.Tensor):
            y_mean = y_mean.item()
        if isinstance(y_std, torch.Tensor):
            y_std = y_std.item()
    except (FileNotFoundError, RuntimeError, KeyError):
        print("File 'demand_model.pth' not found or corrupted. Training a new model.")
        model = train_model()
        checkpoint = torch.load('demand_model.pth', map_location=torch.device('cpu'))
        y_mean, y_std = checkpoint['mean'], checkpoint['std']
        if isinstance(y_mean, torch.Tensor):
            y_mean = y_mean.item()
        if isinstance(y_std, torch.Tensor):
            y_std = y_std.item()

    model.eval()

    if len(historical_data) < 7:
        historical_data = [[0, 0]] * (7 - len(historical_data)) + historical_data
    data = [item for sublist in historical_data[-7:] for item in sublist]
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    data_mean, data_std = data.mean(), data.std()
    data = (data - data_mean) / data_std if data_std != 0 else data

    with torch.no_grad():
        prediction_total = model(data).item()
    prediction_total = prediction_total * y_std + y_mean
    
    # Forecast of order count — average over the last 7 days
    recent_counts = [day[1] for day in historical_data[-7:]]
    prediction_count = sum(recent_counts) / len(recent_counts) if recent_counts else 0
    prediction_count = max(prediction_count, 1)  # Minimum 1 order
    
    # Average check
    prediction_avg = prediction_total / prediction_count if prediction_count > 0 else 0

    if len(historical_data) < 7 or sum(x[0] for x in historical_data) == 0:
        week_avg_total = sum(x[0] for x in historical_data) / len(historical_data) if historical_data else 0
        prediction_total = max(prediction_total, week_avg_total)

    return {
        'total': round(prediction_total, 2),
        'count': round(prediction_count, 2),
        'avg': round(prediction_avg, 2)
    }

if __name__ == "__main__":
    today = timezone.now().date()
    test_data = [
        [100, 2], [120, 3], [130, 4], [110, 2], [150, 5], [140, 4], [160, 3]
    ]
    forecast = predict_demand(test_data)
    print(f"Forecast for tomorrow: {forecast}")
