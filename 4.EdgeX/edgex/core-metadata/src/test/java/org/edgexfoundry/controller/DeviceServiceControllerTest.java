/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.when;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.edgexfoundry.controller.impl.DeviceServiceControllerImpl;
import org.edgexfoundry.dao.AddressableDao;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.DeviceServiceDao;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.dao.ProvisionWatcherRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.AddressableData;
import org.edgexfoundry.test.data.DeviceData;
import org.edgexfoundry.test.data.ServiceData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.data.domain.Sort;

@Category(RequiresNone.class)
public class DeviceServiceControllerTest {

  private static final String LIMIT_PROPERTY = "maxLimit";
  private static final int MAX_LIMIT = 100;

  private static final String TEST_ID = "123";
  private static final String TEST_ERR_MSG = "test message";

  @InjectMocks
  private DeviceServiceControllerImpl controller;

  @Mock
  private DeviceServiceRepository repos;

  @Mock
  private DeviceServiceDao dao;

  @Mock
  private AddressableDao addressableDao;

  @Mock
  private DeviceRepository deviceRepos;

  @Mock
  private ProvisionWatcherRepository watcherRepos;

  private DeviceService service;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    setControllerMAXLIMIT(MAX_LIMIT);
    service = ServiceData.newTestInstance();
    service.setId(TEST_ID);
  }

  @Test
  public void testDeviceService() {
    when(repos.findOne(TEST_ID)).thenReturn(service);
    assertEquals("Device Service returned is not as expected", service,
        controller.deviceService(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceReportNotFound() {
    controller.deviceService(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceReportException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceService(TEST_ID);
  }

  @Test
  public void testDeviceReports() {
    List<DeviceService> srvs = new ArrayList<>();
    srvs.add(service);
    when(repos.findAll(any(Sort.class))).thenReturn(srvs);
    when(repos.count()).thenReturn(1L);
    List<DeviceService> services = controller.deviceServices();
    assertEquals("Number of device services returned does not matched expected number", 1,
        services.size());
    assertEquals("Service returned is not as expected", service, services.get(0));
  }

  @Test(expected = LimitExceededException.class)
  public void testDeviceServicesMaxLimit() {
    List<DeviceService> services = new ArrayList<>();
    services.add(service);
    when(repos.count()).thenReturn(1000L);
    controller.deviceServices();
  }

  @Test(expected = ServiceException.class)
  public void testDeviceServicesException() {
    List<DeviceService> srvs = new ArrayList<>();
    srvs.add(service);
    when(repos.findAll(any(Sort.class))).thenThrow(new RuntimeException(TEST_ERR_MSG));
    when(repos.count()).thenReturn(1L);
    controller.deviceServices();
  }

  @Test
  public void testDeviceServiceForName() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    assertEquals("Device service returned is not as expected", service,
        controller.deviceServiceForName(ServiceData.TEST_SERVICE_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceServiceForNameNotFound() {
    controller.deviceServiceForName(ServiceData.TEST_SERVICE_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceServiceForNameException() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceServiceForName(ServiceData.TEST_SERVICE_NAME);
  }

  @Test
  public void testDeviceServicesForAddressable() {
    Addressable addressable = AddressableData.newTestInstance();
    List<DeviceService> srvs = new ArrayList<>();
    srvs.add(service);
    when(addressableDao.getById(TEST_ID)).thenReturn(addressable);
    when(repos.findByAddressable(addressable)).thenReturn(srvs);
    List<DeviceService> services = controller.deviceServicesForAddressable(TEST_ID);
    assertEquals("Number of services returned does not matched expected number", 1,
        services.size());
    assertEquals("Service returned is not as expected", service, services.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceServiceForAddressableNoneFound() {
    when(addressableDao.getById(TEST_ID)).thenReturn(null);
    controller.deviceServicesForAddressable(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceServiceForAddressableException() {
    when(addressableDao.getById(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceServicesForAddressable(TEST_ID);
  }

  @Test
  public void testDeviceServiceForAddressableByName() {
    Addressable addressable = AddressableData.newTestInstance();
    List<DeviceService> srvs = new ArrayList<>();
    srvs.add(service);
    when(addressableDao.getByName(AddressableData.TEST_ADDR_NAME)).thenReturn(addressable);
    when(repos.findByAddressable(addressable)).thenReturn(srvs);
    List<DeviceService> services =
        controller.deviceServicesForAddressableByName(AddressableData.TEST_ADDR_NAME);
    assertEquals("Number of services returned does not matched expected number", 1,
        services.size());
    assertEquals("Service returned is not as expected", service, services.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceServiceForAddressableByNameNoneFound() {
    when(addressableDao.getByName(AddressableData.TEST_ADDR_NAME)).thenReturn(null);
    controller.deviceServicesForAddressableByName(AddressableData.TEST_ADDR_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceServiceForAddressableByNameException() {
    when(addressableDao.getByName(AddressableData.TEST_ADDR_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceServicesForAddressableByName(AddressableData.TEST_ADDR_NAME);
  }

  @Test
  public void testDeviceServicesByLabel() {
    List<DeviceService> srvs = new ArrayList<>();
    srvs.add(service);
    when(repos.findByLabelsIn(ServiceData.TEST_LABELS[0])).thenReturn(srvs);
    List<DeviceService> services = controller.deviceServicesByLabel(ServiceData.TEST_LABELS[0]);
    assertEquals("Number of services returned does not matched expected number", 1,
        services.size());
    assertEquals("Service returned is not as expected", service, services.get(0));
  }

  @Test(expected = ServiceException.class)
  public void testDeviceServiceByLabelException() {
    when(repos.findByLabelsIn(ServiceData.TEST_LABELS[0]))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceServicesByLabel(ServiceData.TEST_LABELS[0]);
  }

  @Test
  public void testAddressablesForAssociatedDevices() {
    Addressable addressable = AddressableData.newTestInstance();
    Device device = DeviceData.newTestInstance();
    device.setAddressable(addressable);
    List<Device> devices = new ArrayList<>();
    devices.add(device);
    when(repos.findOne(TEST_ID)).thenReturn(service);
    when(deviceRepos.findByService(service)).thenReturn(devices);
    Set<Addressable> addressables = controller.addressablesForAssociatedDevices(TEST_ID);
    assertEquals("Number of addressables returned does not matched expected number", 1,
        addressables.size());
    assertTrue("Addressable returned is not as expected", addressables.contains(addressable));
  }

  @Test(expected = NotFoundException.class)
  public void testAddressablesForAssociatedDevicesNullService() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.addressablesForAssociatedDevices(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testAddressablesForAssociatedDevicesException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.addressablesForAssociatedDevices(TEST_ID);
  }

  @Test
  public void testAddressablesForAssociatedDevicesByName() {
    Addressable addressable = AddressableData.newTestInstance();
    Device device = DeviceData.newTestInstance();
    device.setAddressable(addressable);
    List<Device> devices = new ArrayList<>();
    devices.add(device);
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    when(deviceRepos.findByService(service)).thenReturn(devices);
    Set<Addressable> addressables =
        controller.addressablesForAssociatedDevicesByName(ServiceData.TEST_SERVICE_NAME);
    assertEquals("Number of addressables returned does not matched expected number", 1,
        addressables.size());
    assertTrue("Addressable returned is not as expected", addressables.contains(addressable));
  }

  @Test(expected = NotFoundException.class)
  public void testAddressablesForAssociatedDevicesByNameNullService() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(null);
    controller.addressablesForAssociatedDevicesByName(ServiceData.TEST_SERVICE_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testAddressablesForAssociatedDevicesByNameException() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.addressablesForAssociatedDevicesByName(ServiceData.TEST_SERVICE_NAME);
  }

  @Test
  public void testAdd() {
    Addressable addressable = AddressableData.newTestInstance();
    service.setAddressable(addressable);
    when(addressableDao.getByIdOrName(addressable)).thenReturn(addressable);
    controller.add(service);
  }

  @Test(expected = ServiceException.class)
  public void testAddWithNullDeviceService() {
    controller.add(null);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithDuplicatKey() {
    Addressable addressable = AddressableData.newTestInstance();
    service.setAddressable(addressable);
    when(addressableDao.getByIdOrName(addressable)).thenReturn(addressable);
    when(repos.save(service)).thenThrow(new DuplicateKeyException(TEST_ERR_MSG));
    controller.add(service);
  }

  @Test(expected = ServiceException.class)
  public void testAddException() {
    Addressable addressable = AddressableData.newTestInstance();
    service.setAddressable(addressable);
    when(addressableDao.getByIdOrName(addressable)).thenReturn(addressable);
    when(repos.save(service)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.add(service);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithNotFoundAddressable() {
    Addressable addressable = AddressableData.newTestInstance();
    service.setAddressable(addressable);
    when(addressableDao.getByIdOrName(addressable)).thenReturn(null);
    controller.add(service);
  }

  @Test(expected = ServiceException.class)
  public void testAddAddressableDaoException() {
    Addressable addressable = AddressableData.newTestInstance();
    service.setAddressable(addressable);
    when(addressableDao.getByIdOrName(addressable)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.add(service);
  }

  @Test
  public void testUpdateLastConnected() {
    when(repos.findOne(TEST_ID)).thenReturn(service);
    assertTrue("Device service connected time was not updated",
        controller.updateLastConnected(TEST_ID, System.currentTimeMillis()));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateLastConnectedNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.updateLastConnected(TEST_ID, System.currentTimeMillis());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastConnectedException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateLastConnected(TEST_ID, System.currentTimeMillis());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastConnectedDaoException() {
    when(repos.findOne(TEST_ID)).thenReturn(service);
    when(repos.save(service)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateLastConnected(TEST_ID, System.currentTimeMillis());
  }

  @Test
  public void testUpdateLastConnectedByName() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    assertTrue("Device connected time was not updated", controller
        .updateLastConnectedByName(ServiceData.TEST_SERVICE_NAME, System.currentTimeMillis()));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateLastConnectedByNameNotFound() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(null);
    controller.updateLastConnectedByName(ServiceData.TEST_SERVICE_NAME, System.currentTimeMillis());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastConnectedByNameException() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateLastConnectedByName(ServiceData.TEST_SERVICE_NAME, System.currentTimeMillis());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastConnectedByNameDaoException() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    when(repos.save(service)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateLastConnectedByName(ServiceData.TEST_SERVICE_NAME, System.currentTimeMillis());
  }

  @Test
  public void testUpdateLastReported() {
    when(repos.findOne(TEST_ID)).thenReturn(service);
    assertTrue("Device service reported time was not updated",
        controller.updateLastReported(TEST_ID, System.currentTimeMillis()));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateLastReportedNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    assertTrue("Device reported time was not updated",
        controller.updateLastReported(TEST_ID, System.currentTimeMillis()));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastReportedException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    assertTrue("Device service reported time was not updated",
        controller.updateLastReported(TEST_ID, System.currentTimeMillis()));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastReportedDaoException() {
    when(repos.findOne(TEST_ID)).thenReturn(service);
    when(repos.save(service)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateLastReported(TEST_ID, System.currentTimeMillis());
  }

  @Test
  public void testUpdateLastReportedByName() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    assertTrue("Device service reported time was not updated", controller
        .updateLastReportedByName(ServiceData.TEST_SERVICE_NAME, System.currentTimeMillis()));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateLastReportedByNameNotFound() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(null);
    assertTrue("Device service reported time was not updated", controller
        .updateLastReportedByName(ServiceData.TEST_SERVICE_NAME, System.currentTimeMillis()));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastReportedByNameException() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    assertTrue("Device service reported time was not updated", controller
        .updateLastReportedByName(ServiceData.TEST_SERVICE_NAME, System.currentTimeMillis()));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastReportedByNameDaoException() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    when(repos.save(service)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateLastReportedByName(ServiceData.TEST_SERVICE_NAME, System.currentTimeMillis());
  }

  @Test
  public void testUpdateOpState() {
    when(repos.findOne(TEST_ID)).thenReturn(service);
    assertTrue("Device service op state was not updated",
        controller.updateOpState(TEST_ID, ServiceData.TEST_OP.toString()));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateOpStateNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.updateOpState(TEST_ID, ServiceData.TEST_OP.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateOpStateException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateOpState(TEST_ID, ServiceData.TEST_OP.toString());
  }

  @Test
  public void testUpdateOpStateByName() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    assertTrue("Device service op state was not updated", controller
        .updateOpStateByName(ServiceData.TEST_SERVICE_NAME, DeviceData.TEST_OP.toString()));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateOpStateByNameNotFound() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(null);
    controller.updateOpStateByName(ServiceData.TEST_SERVICE_NAME, DeviceData.TEST_OP.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateOpStateByNameException() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateOpStateByName(ServiceData.TEST_SERVICE_NAME, DeviceData.TEST_OP.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateOpStateByNameExceptionInSave() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    when(repos.save(service)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateOpStateByName(ServiceData.TEST_SERVICE_NAME, DeviceData.TEST_OP.toString());
  }

  @Test
  public void testUpdateAdminState() {
    when(repos.findOne(TEST_ID)).thenReturn(service);
    assertTrue("Device service admin state was not updated",
        controller.updateAdminState(TEST_ID, DeviceData.TEST_ADMIN.toString()));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateAdminStateNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.updateAdminState(TEST_ID, DeviceData.TEST_ADMIN.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateAdminStateException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateAdminState(TEST_ID, DeviceData.TEST_ADMIN.toString());
  }

  @Test
  public void testUpdateAdminStateByName() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    assertTrue("Device service admin state was not updated", controller
        .updateAdminStateByName(ServiceData.TEST_SERVICE_NAME, DeviceData.TEST_ADMIN.toString()));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateAdminStateByNameNotFound() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(null);
    controller.updateAdminStateByName(ServiceData.TEST_SERVICE_NAME,
        DeviceData.TEST_ADMIN.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateAdminStateByNameException() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateAdminStateByName(ServiceData.TEST_SERVICE_NAME,
        DeviceData.TEST_ADMIN.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateAdminStateByNameExceptionInSave() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    when(repos.save(service)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateAdminStateByName(ServiceData.TEST_SERVICE_NAME,
        DeviceData.TEST_ADMIN.toString());
  }

  @Test
  public void testUpdate() {
    when(dao.getByIdOrName(service)).thenReturn(service);
    assertTrue("Device service was not updated", controller.update(service));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateWithNull() {
    controller.update(null);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithNoServiceID() {
    service.setId(null);
    when(dao.getByIdOrName(service)).thenReturn(null);
    controller.update(service);
  }

  @Test(expected = ServiceException.class)
  public void testUpdateException() {
    when(dao.getByIdOrName(service)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.update(service);
  }

  @Test
  public void testUpdateWithAddressable() {
    Addressable addressable = AddressableData.newTestInstance();
    service.setAddressable(addressable);
    when(addressableDao.getByIdOrName(addressable)).thenReturn(addressable);
    when(dao.getByIdOrName(service)).thenReturn(service);
    assertTrue("Device service was not updated", controller.update(service));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithAddressableNotFound() {
    Addressable addressable = AddressableData.newTestInstance();
    service.setAddressable(addressable);
    when(addressableDao.getByIdOrName(addressable)).thenReturn(null);
    when(dao.getByIdOrName(service)).thenReturn(service);
    controller.update(service);
  }

  @Test
  public void testDelete() {
    when(repos.findOne(TEST_ID)).thenReturn(service);
    assertTrue("Device service was not deleted", controller.delete(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.delete(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    assertTrue("Device service was not deleted", controller.delete(TEST_ID));
  }

  @Test
  public void testDeleteByName() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    assertTrue("Device service was not deleted",
        controller.deleteByName(ServiceData.TEST_SERVICE_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameNotFound() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(null);
    controller.deleteByName(ServiceData.TEST_SERVICE_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameException() {
    when(repos.findByName(ServiceData.TEST_SERVICE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    assertTrue("Device service was not deleted",
        controller.deleteByName(ServiceData.TEST_SERVICE_NAME));
  }

  private void setControllerMAXLIMIT(int newLimit) throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT_PROPERTY);
    temp.setAccessible(true);
    temp.set(controller, newLimit);
  }
}
