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
 * @microservice: support-logging
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

import org.edgexfoundry.controller.impl.CallbackExecutor;
import org.edgexfoundry.controller.impl.DeviceControllerImpl;
import org.edgexfoundry.dao.AddressableDao;
import org.edgexfoundry.dao.DeviceProfileDao;
import org.edgexfoundry.dao.DeviceReportDao;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.DeviceServiceDao;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.AddressableData;
import org.edgexfoundry.test.data.DeviceData;
import org.edgexfoundry.test.data.ProfileData;
import org.edgexfoundry.test.data.ServiceData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.data.domain.Sort;

@Category(RequiresNone.class)
public class DeviceControllerTest {

  private static final String LIMIT_PROPERTY = "maxLimit";
  private static final int MAX_LIMIT = 100;

  private static final String TEST_ID = "123";
  private static final String TEST_ERR_MSG = "test message";

  @InjectMocks
  private DeviceControllerImpl controller;

  @Mock
  private DeviceRepository repos;

  @Mock
  private AddressableDao addressableDao;

  @Mock
  private DeviceProfileDao profileDao;

  @Mock
  private DeviceServiceDao serviceDao;

  @Mock
  private CallbackExecutor callback;

  @Mock
  private DeviceReportDao deviceRptDao;

  @Mock
  private NotificationClient notificationClient;

  private Device device;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    setControllerMAXLIMIT(MAX_LIMIT);
    device = DeviceData.newTestInstance();
    device.setId(TEST_ID);
  }

  @Test
  public void testDevice() {
    when(repos.findOne(TEST_ID)).thenReturn(device);
    assertEquals("Device returned is not as expected", device, controller.device(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceNotFound() {
    controller.device(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.device(TEST_ID);
  }

  @Test
  public void testDevices() {
    List<Device> devs = new ArrayList<>();
    devs.add(device);
    when(repos.findAll(any(Sort.class))).thenReturn(devs);
    when(repos.count()).thenReturn(1L);
    List<Device> devices = controller.devices();
    assertEquals("Number of devices returned does not matched expected number", 1, devices.size());
    assertEquals("Device returned is not as expected", device, devices.get(0));
  }

  @Test(expected = LimitExceededException.class)
  public void testDevicesMaxLimit() {
    List<Device> devs = new ArrayList<>();
    devs.add(device);
    when(repos.count()).thenReturn(1000L);
    controller.devices();
  }

  @Test(expected = ServiceException.class)
  public void testDevicesException() {
    List<Device> devs = new ArrayList<>();
    devs.add(device);
    when(repos.findAll(any(Sort.class))).thenThrow(new RuntimeException(TEST_ERR_MSG));
    when(repos.count()).thenReturn(1L);
    controller.devices();
  }

  @Test
  public void testDeviceForName() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    assertEquals("Device returned is not as expected", device,
        controller.deviceForName(DeviceData.TEST_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceForNameNotFound() {
    controller.deviceForName(DeviceData.TEST_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceForNameException() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceForName(DeviceData.TEST_NAME);
  }

  @Test
  public void testDeviceByLabel() {
    List<Device> devs = new ArrayList<>();
    devs.add(device);
    when(repos.findByLabelsIn(DeviceData.TEST_LABELS[0])).thenReturn(devs);
    List<Device> devices = controller.devicesByLabel(DeviceData.TEST_LABELS[0]);
    assertEquals("Number of devices returned does not matched expected number", 1, devices.size());
    assertEquals("Device returned is not as expected", device, devices.get(0));
  }

  @Test(expected = ServiceException.class)
  public void testDeviceByLabelException() {
    when(repos.findByLabelsIn(DeviceData.TEST_LABELS[0]))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.devicesByLabel(DeviceData.TEST_LABELS[0]);
  }

  @Test
  public void testDeviceByService() {
    DeviceService service = new DeviceService();
    List<Device> devs = new ArrayList<>();
    devs.add(device);
    when(serviceDao.getById(TEST_ID)).thenReturn(service);
    when(repos.findByService(service)).thenReturn(devs);
    List<Device> devices = controller.devicesForService(TEST_ID);
    assertEquals("Number of devices returned does not matched expected number", 1, devices.size());
    assertEquals("Device returned is not as expected", device, devices.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceByServiceNoneFound() {
    when(serviceDao.getById(TEST_ID)).thenReturn(null);
    controller.devicesForService(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceByServiceException() {
    when(serviceDao.getById(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.devicesForService(TEST_ID);
  }

  @Test
  public void testDeviceByServiceName() {
    DeviceService service = new DeviceService();
    List<Device> devs = new ArrayList<>();
    devs.add(device);
    when(serviceDao.getByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(service);
    when(repos.findByService(service)).thenReturn(devs);
    List<Device> devices = controller.devicesForServiceByName(ServiceData.TEST_SERVICE_NAME);
    assertEquals("Number of devices returned does not matched expected number", 1, devices.size());
    assertEquals("Device returned is not as expected", device, devices.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceByServiceNameNoneFound() {
    when(serviceDao.getByName(ServiceData.TEST_SERVICE_NAME)).thenReturn(null);
    controller.devicesForServiceByName(ServiceData.TEST_SERVICE_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceByServiceNameException() {
    when(serviceDao.getByName(ServiceData.TEST_SERVICE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.devicesForServiceByName(ServiceData.TEST_SERVICE_NAME);
  }

  @Test
  public void testDeviceForProfile() {
    DeviceProfile profile = new DeviceProfile();
    List<Device> devs = new ArrayList<>();
    devs.add(device);
    when(profileDao.getById(TEST_ID)).thenReturn(profile);
    when(repos.findByProfile(profile)).thenReturn(devs);
    List<Device> devices = controller.devicesForProfile(TEST_ID);
    assertEquals("Number of devices returned does not matched expected number", 1, devices.size());
    assertEquals("Device returned is not as expected", device, devices.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceForProfileNoneFound() {
    when(profileDao.getById(TEST_ID)).thenReturn(null);
    controller.devicesForProfile(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceForProfileException() {
    when(profileDao.getById(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.devicesForProfile(TEST_ID);
  }

  @Test
  public void testDeviceForProfileByName() {
    DeviceProfile profile = new DeviceProfile();
    List<Device> devs = new ArrayList<>();
    devs.add(device);
    when(profileDao.getByName(ProfileData.TEST_PROFILE_NAME)).thenReturn(profile);
    when(repos.findByProfile(profile)).thenReturn(devs);
    List<Device> devices = controller.devicesForProfileByName(ProfileData.TEST_PROFILE_NAME);
    assertEquals("Number of devices returned does not matched expected number", 1, devices.size());
    assertEquals("Device returned is not as expected", device, devices.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceForProfileByNameNoneFound() {
    when(profileDao.getByName(ProfileData.TEST_PROFILE_NAME)).thenReturn(null);
    controller.devicesForProfileByName(ProfileData.TEST_PROFILE_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceForProfileByNameException() {
    when(profileDao.getByName(ProfileData.TEST_PROFILE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.devicesForProfileByName(ProfileData.TEST_PROFILE_NAME);
  }

  @Test
  public void testDeviceForAddressable() {
    Addressable addressable = AddressableData.newTestInstance();
    List<Device> devs = new ArrayList<>();
    devs.add(device);
    when(addressableDao.getById(TEST_ID)).thenReturn(addressable);
    when(repos.findByAddressable(addressable)).thenReturn(devs);
    List<Device> devices = controller.devicesForAddressable(TEST_ID);
    assertEquals("Number of devices returned does not matched expected number", 1, devices.size());
    assertEquals("Device returned is not as expected", device, devices.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceForAddressableNoneFound() {
    when(addressableDao.getById(TEST_ID)).thenReturn(null);
    controller.devicesForAddressable(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceForAddressableException() {
    when(addressableDao.getById(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.devicesForAddressable(TEST_ID);
  }

  @Test
  public void testDeviceForAddressableByName() {
    Addressable addressable = AddressableData.newTestInstance();
    List<Device> devs = new ArrayList<>();
    devs.add(device);
    when(addressableDao.getByName(AddressableData.TEST_ADDR_NAME)).thenReturn(addressable);
    when(repos.findByAddressable(addressable)).thenReturn(devs);
    List<Device> devices = controller.devicesForAddressableByName(AddressableData.TEST_ADDR_NAME);
    assertEquals("Number of devices returned does not matched expected number", 1, devices.size());
    assertEquals("Device returned is not as expected", device, devices.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceForAddressableByNameNoneFound() {
    when(addressableDao.getByName(AddressableData.TEST_ADDR_NAME)).thenReturn(null);
    controller.devicesForAddressableByName(AddressableData.TEST_ADDR_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceForAddressableByNameException() {
    when(addressableDao.getByName(AddressableData.TEST_ADDR_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.devicesForAddressableByName(AddressableData.TEST_ADDR_NAME);
  }

  @Test
  public void testAdd() {
    DeviceService service = ServiceData.newTestInstance();
    DeviceProfile profile = ProfileData.newTestInstance();
    Addressable addressable = AddressableData.newTestInstance();
    device.setService(service);
    device.setProfile(profile);
    device.setAddressable(addressable);
    when(serviceDao.getByIdOrName(service)).thenReturn(service);
    when(profileDao.getByIdOrName(profile)).thenReturn(profile);
    when(addressableDao.getByIdOrName(addressable)).thenReturn(addressable);
    when(repos.save(device)).thenReturn(device);
    assertEquals("Device ID returned is not the value expected", TEST_ID, controller.add(device));
  }

  @Test(expected = ServiceException.class)
  public void testAddWithNull() {
    controller.add(null);
  }

  @Test(expected = ServiceException.class)
  public void testAddServiceException() {
    DeviceService service = ServiceData.newTestInstance();
    DeviceProfile profile = ProfileData.newTestInstance();
    Addressable addressable = AddressableData.newTestInstance();
    device.setService(service);
    device.setProfile(profile);
    device.setAddressable(addressable);
    when(serviceDao.getByIdOrName(service)).thenReturn(service);
    when(profileDao.getByIdOrName(profile)).thenReturn(profile);
    when(addressableDao.getByIdOrName(addressable)).thenReturn(addressable);
    when(repos.save(device)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.add(device);
  }

  @Test(expected = DataValidationException.class)
  public void testAddNoDeviceServiceAssociated() {
    DeviceService service = ServiceData.newTestInstance();
    DeviceProfile profile = ProfileData.newTestInstance();
    Addressable addressable = AddressableData.newTestInstance();
    device.setService(service);
    device.setProfile(profile);
    device.setAddressable(addressable);
    when(serviceDao.getByIdOrName(service)).thenReturn(null);
    when(profileDao.getByIdOrName(profile)).thenReturn(profile);
    when(addressableDao.getByIdOrName(addressable)).thenReturn(addressable);
    when(repos.save(device)).thenReturn(device);
    controller.add(device);
  }

  @Test(expected = DataValidationException.class)
  public void testAddNoDeviceProfileAssociated() {
    DeviceService service = ServiceData.newTestInstance();
    DeviceProfile profile = ProfileData.newTestInstance();
    Addressable addressable = AddressableData.newTestInstance();
    device.setService(service);
    device.setProfile(profile);
    device.setAddressable(addressable);
    when(serviceDao.getByIdOrName(service)).thenReturn(service);
    when(profileDao.getByIdOrName(profile)).thenReturn(null);
    when(addressableDao.getByIdOrName(addressable)).thenReturn(addressable);
    when(repos.save(device)).thenReturn(device);
    controller.add(device);
  }

  @Test(expected = DataValidationException.class)
  public void testAddNoAddressableAssociated() {
    DeviceService service = ServiceData.newTestInstance();
    DeviceProfile profile = ProfileData.newTestInstance();
    Addressable addressable = AddressableData.newTestInstance();
    device.setService(service);
    device.setProfile(profile);
    device.setAddressable(addressable);
    when(serviceDao.getByIdOrName(service)).thenReturn(service);
    when(profileDao.getByIdOrName(profile)).thenReturn(profile);
    when(addressableDao.getByIdOrName(addressable)).thenReturn(null);
    when(repos.save(device)).thenReturn(device);
    controller.add(device);
  }

  @Test
  public void testUpdateLastConnected() {
    when(repos.findOne(TEST_ID)).thenReturn(device);
    assertTrue("Device connected time was not updated",
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

  @Test
  public void testUpdateLastConnectedWithNotify() {
    when(repos.findOne(TEST_ID)).thenReturn(device);
    assertTrue("Device connected time was not updated",
        controller.updateLastConnected(TEST_ID, System.currentTimeMillis(), true));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastConnectedWithNotifyException() {
    when(repos.findOne(TEST_ID)).thenReturn(device);
    when(repos.save(device)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateLastConnected(TEST_ID, System.currentTimeMillis(), true);
  }

  @Test
  public void testUpdateLastConnectedByName() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    assertTrue("Device connected time was not updated",
        controller.updateLastConnectedByName(DeviceData.TEST_NAME, System.currentTimeMillis()));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateLastConnectedByNameNotFound() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(null);
    controller.updateLastConnectedByName(DeviceData.TEST_NAME, System.currentTimeMillis());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastConnectedByNameException() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateLastConnectedByName(DeviceData.TEST_NAME, System.currentTimeMillis());
  }

  @Test
  public void testUpdateLastConnectedByNameWithNotify() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    assertTrue("Device connected time was not updated", controller
        .updateLastConnectedByName(DeviceData.TEST_NAME, System.currentTimeMillis(), true));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastConnectedByNameWithNotifyException() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    when(repos.save(device)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    assertTrue("Device connected time was not updated", controller
        .updateLastConnectedByName(DeviceData.TEST_NAME, System.currentTimeMillis(), true));
  }

  @Test
  public void testUpdateLastReported() {
    when(repos.findOne(TEST_ID)).thenReturn(device);
    assertTrue("Device reported time was not updated",
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
    assertTrue("Device reported time was not updated",
        controller.updateLastReported(TEST_ID, System.currentTimeMillis()));
  }

  @Test
  public void testUpdateLastReportedWithNotify() {
    when(repos.findOne(TEST_ID)).thenReturn(device);
    assertTrue("Device reported time was not updated",
        controller.updateLastReported(TEST_ID, System.currentTimeMillis(), true));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastReportedWithNotifyException() {
    when(repos.findOne(TEST_ID)).thenReturn(device);
    when(repos.save(device)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    assertTrue("Device reported time was not updated",
        controller.updateLastReported(TEST_ID, System.currentTimeMillis(), true));
  }

  @Test
  public void testUpdateLastReportedByName() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    assertTrue("Device reported time was not updated",
        controller.updateLastReportedByName(DeviceData.TEST_NAME, System.currentTimeMillis()));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateLastReportedByNameNotFound() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(null);
    assertTrue("Device reported time was not updated",
        controller.updateLastReportedByName(DeviceData.TEST_NAME, System.currentTimeMillis()));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastReportedByNameException() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    assertTrue("Device reported time was not updated",
        controller.updateLastReportedByName(DeviceData.TEST_NAME, System.currentTimeMillis()));
  }

  @Test
  public void testUpdateLastReportedByNameWithNotify() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    assertTrue("Device reported time was not updated", controller
        .updateLastReportedByName(DeviceData.TEST_NAME, System.currentTimeMillis(), true));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateLastReportedByNameWithNotifyException() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    when(repos.save(device)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateLastReportedByName(DeviceData.TEST_NAME, System.currentTimeMillis(), true);
  }

  @Test
  public void testUpdateOpState() {
    when(repos.findOne(TEST_ID)).thenReturn(device);
    assertTrue("Device op state was not updated",
        controller.updateOpState(TEST_ID, DeviceData.TEST_OP.toString()));
  }

  @Test(expected = DataValidationException.class)
  public void testUpdateOpStateWithNull() {
    when(repos.findOne(TEST_ID)).thenReturn(device);
    controller.updateOpState(TEST_ID, null);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateOpStateNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.updateOpState(TEST_ID, DeviceData.TEST_OP.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateOpStateException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateOpState(TEST_ID, DeviceData.TEST_OP.toString());
  }

  @Test
  public void testUpdateOpStateByName() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    assertTrue("Device op state was not updated",
        controller.updateOpStateByName(DeviceData.TEST_NAME, DeviceData.TEST_OP.toString()));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateOpStateByNameNotFound() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(null);
    controller.updateOpStateByName(DeviceData.TEST_NAME, DeviceData.TEST_OP.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateOpStateByNameException() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateOpStateByName(DeviceData.TEST_NAME, DeviceData.TEST_OP.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateOpStateByNameExceptionInSave() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    when(repos.save(device)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateOpStateByName(DeviceData.TEST_NAME, DeviceData.TEST_OP.toString());
  }

  @Test
  public void testUpdateAdminState() {
    when(repos.findOne(TEST_ID)).thenReturn(device);
    assertTrue("Device admin state was not updated",
        controller.updateAdminState(TEST_ID, DeviceData.TEST_ADMIN.toString()));
  }

  @Test(expected = DataValidationException.class)
  public void testUpdateAdminStateWithNull() {
    when(repos.findOne(TEST_ID)).thenReturn(device);
    controller.updateAdminState(TEST_ID, null);
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
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    assertTrue("Device admin state was not updated",
        controller.updateAdminStateByName(DeviceData.TEST_NAME, DeviceData.TEST_ADMIN.toString()));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateAdminStateByNameNotFound() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(null);
    controller.updateAdminStateByName(DeviceData.TEST_NAME, DeviceData.TEST_ADMIN.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateAdminStateByNameException() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateAdminStateByName(DeviceData.TEST_NAME, DeviceData.TEST_ADMIN.toString());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateAdminStateByNameExceptionInSave() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    when(repos.save(device)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.updateAdminStateByName(DeviceData.TEST_NAME, DeviceData.TEST_ADMIN.toString());
  }

  @Test
  public void testUpdate() {
    when(repos.findOne(TEST_ID)).thenReturn(device);
    assertTrue("Device was not updated", controller.update(device));
  }

  @Test
  public void testUpdateWithNoDeviceID() {
    device.setId(null);
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    assertTrue("Device was not updated", controller.update(device));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateWithNull() {
    controller.update(null);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithUnknownDevice() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.update(device);
  }

  @Test(expected = ServiceException.class)
  public void testUpdatException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.update(device);
  }

  @Test
  public void testDelete() {
    when(repos.findOne(TEST_ID)).thenReturn(device);
    assertTrue("Device was not deleted", controller.delete(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.delete(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteDaoFails() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.delete(TEST_ID);
  }

  @Test
  public void testDeleteByName() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(device);
    assertTrue("Device was not deleted", controller.deleteByName(DeviceData.TEST_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameNotFound() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenReturn(null);
    controller.deleteByName(DeviceData.TEST_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameDaoFails() {
    when(repos.findByName(DeviceData.TEST_NAME)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deleteByName(DeviceData.TEST_NAME);
  }

  private void setControllerMAXLIMIT(int newLimit) throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT_PROPERTY);
    temp.setAccessible(true);
    temp.set(controller, newLimit);
  }
}
